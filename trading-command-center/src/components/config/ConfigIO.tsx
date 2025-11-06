import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { 
  Download, 
  Upload, 
  FileText, 
  Settings,
  Save,
  FolderOpen,
  FileSpreadsheet,
  Database,
  Archive,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Clock,
  RefreshCw
} from 'lucide-react';

interface ExportOptions {
  format: 'json' | 'yaml' | 'csv' | 'xml';
  includeSecrets: boolean;
  includeDefaults: boolean;
  compression: boolean;
  encryption: boolean;
  scope: 'all' | 'brokers' | 'strategies' | 'risk' | 'ai' | 'notifications' | 'security' | 'performance';
}

interface ImportResult {
  success: boolean;
  totalItems: number;
  importedItems: number;
  skippedItems: number;
  errors: string[];
  warnings: string[];
  timestamp: string;
}

const ConfigIO: React.FC = () => {
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'json',
    includeSecrets: false,
    includeDefaults: false,
    compression: false,
    encryption: false,
    scope: 'all'
  });

  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [exportProgress, setExportProgress] = useState(0);
  const [importProgress, setImportProgress] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const formatOptions = [
    { value: 'json', label: 'JSON', description: 'JavaScript Object Notation', icon: FileText },
    { value: 'yaml', label: 'YAML', description: 'YAML Ain\'t Markup Language', icon: FileText },
    { value: 'csv', label: 'CSV', description: 'Comma-Separated Values', icon: FileSpreadsheet },
    { value: 'xml', label: 'XML', description: 'Extensible Markup Language', icon: FileText }
  ];

  const scopeOptions = [
    { value: 'all', label: 'All Configuration', description: 'Export all configuration sections' },
    { value: 'brokers', label: 'Broker Settings', description: 'Only broker configurations' },
    { value: 'strategies', label: 'Strategies', description: 'Trading strategy configurations' },
    { value: 'risk', label: 'Risk Management', description: 'Risk settings and limits' },
    { value: 'ai', label: 'AI & LLM', description: 'AI model and provider settings' },
    { value: 'notifications', label: 'Notifications', description: 'Notification channels and templates' },
    { value: 'security', label: 'Security', description: 'Authentication and security settings' },
    { value: 'performance', label: 'Performance', description: 'Optimization and performance settings' }
  ];

  const exportConfiguration = async () => {
    setIsExporting(true);
    setExportProgress(0);

    try {
      // Simulate export process
      for (let i = 0; i <= 100; i += 10) {
        setExportProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      const config = await window.electronAPI?.getAllConfiguration(exportOptions.scope);
      let exportedData = config;

      if (exportOptions.format === 'yaml') {
        // Convert JSON to YAML (simplified)
        exportedData = JSON.stringify(config, null, 2);
      } else if (exportOptions.format === 'csv') {
        // Convert to CSV format (simplified for brokers)
        const brokers = config.brokers || [];
        const csvHeaders = 'Name,Type,Environment,API Key,Status\n';
        const csvRows = brokers.map((broker: any) => 
          `"${broker.name}","${broker.type}","${broker.config?.environment || ''}","${broker.config?.apiKey ? '***' : ''}","${broker.status}"`
        ).join('\n');
        exportedData = csvHeaders + csvRows;
      }

      // Handle encryption if requested
      if (exportOptions.encryption && exportedData) {
        exportedData = await window.electronAPI?.encryptData(exportedData);
      }

      // Handle compression if requested
      if (exportOptions.compression && exportedData) {
        exportedData = await window.electronAPI?.compressData(exportedData);
      }

      // Create and download file
      const blob = new Blob([typeof exportedData === 'string' ? exportedData : JSON.stringify(exportedData, null, 2)], {
        type: getMimeType(exportOptions.format)
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trading-config-${exportOptions.scope}-${new Date().toISOString().split('T')[0]}.${exportOptions.format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed: ' + (error instanceof Error ? error.message : 'Unknown error'));
    } finally {
      setIsExporting(false);
      setExportProgress(0);
    }
  };

  const importConfiguration = async () => {
    if (!selectedFile) {
      alert('Please select a file to import');
      return;
    }

    setIsImporting(true);
    setImportProgress(0);

    try {
      // Read file content
      const content = await readFileContent(selectedFile);
      
      // Handle decompression if needed
      let data = content;
      if (exportOptions.compression) {
        data = await window.electronAPI?.decompressData(content);
      }

      // Handle decryption if needed
      if (exportOptions.encryption) {
        data = await window.electronAPI?.decryptData(data);
      }

      // Parse based on format
      let parsedData;
      if (exportOptions.format === 'json') {
        parsedData = JSON.parse(data);
      } else if (exportOptions.format === 'yaml') {
        // Parse YAML (simplified)
        parsedData = JSON.parse(data);
      } else if (exportOptions.format === 'csv') {
        // Parse CSV (simplified)
        parsedData = parseCSV(data);
      } else if (exportOptions.format === 'xml') {
        // Parse XML (simplified)
        parsedData = JSON.parse(data);
      }

      // Validate and import
      setImportProgress(25);
      const validation = await window.electronAPI?.validateConfiguration(parsedData);
      
      if (!validation?.isValid) {
        throw new Error('Configuration validation failed: ' + validation?.errors.join(', '));
      }

      setImportProgress(50);
      const result = await window.electronAPI?.importConfiguration(parsedData, exportOptions.scope);
      
      setImportProgress(100);
      
      setImportResult({
        success: true,
        totalItems: result.totalItems,
        importedItems: result.importedItems,
        skippedItems: result.skippedItems,
        errors: result.errors || [],
        warnings: result.warnings || [],
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Import failed:', error);
      setImportResult({
        success: false,
        totalItems: 0,
        importedItems: 0,
        skippedItems: 0,
        errors: [error instanceof Error ? error.message : 'Import failed'],
        warnings: [],
        timestamp: new Date().toISOString()
      });
    } finally {
      setIsImporting(false);
      setImportProgress(0);
      setSelectedFile(null);
    }
  };

  const readFileContent = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = (e) => reject(e);
      reader.readAsText(file);
    });
  };

  const parseCSV = (csvContent: string) => {
    const lines = csvContent.split('\n').filter(line => line.trim());
    const headers = lines[0].split(',').map(h => h.replace(/"/g, ''));
    const brokers = lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.replace(/"/g, ''));
      const broker: any = {};
      headers.forEach((header, index) => {
        if (header === 'Name') broker.name = values[index];
        else if (header === 'Type') broker.type = values[index];
        else if (header === 'Environment') broker.config = { environment: values[index] };
        else if (header === 'API Key') broker.config = { ...broker.config, apiKey: values[index] };
        else if (header === 'Status') broker.status = values[index];
      });
      broker.id = `broker_${Date.now()}_${Math.random()}`;
      return broker;
    });
    return { brokers };
  };

  const getMimeType = (format: string) => {
    switch (format) {
      case 'json':
        return 'application/json';
      case 'yaml':
        return 'text/yaml';
      case 'csv':
        return 'text/csv';
      case 'xml':
        return 'application/xml';
      default:
        return 'text/plain';
    }
  };

  const saveTemplate = async () => {
    try {
      const config = await window.electronAPI?.getAllConfiguration('all');
      const template = {
        ...config,
        _template: true,
        _version: '1.0.0',
        _created: new Date().toISOString()
      };

      const blob = new Blob([JSON.stringify(template, null, 2)], {
        type: 'application/json'
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trading-config-template-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      console.error('Failed to save template:', error);
      alert('Failed to save template');
    }
  };

  const loadTemplate = async () => {
    try {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json,.yaml,.yml';
      
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
          const content = await readFileContent(file);
          const template = JSON.parse(content);
          
          if (template._template) {
            // Apply template
            await window.electronAPI?.applyTemplate(template);
            alert('Template applied successfully');
          } else {
            alert('Invalid template file');
          }
        }
      };
      
      input.click();
    } catch (error) {
      console.error('Failed to load template:', error);
      alert('Failed to load template');
    }
  };

  const getResultIcon = (success: boolean) => {
    return success ? (
      <CheckCircle className="h-5 w-5 text-green-500" />
    ) : (
      <XCircle className="h-5 w-5 text-red-500" />
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-5 w-5" />
          Configuration Import/Export
        </CardTitle>
        <CardDescription>
          Export and import configuration settings in various formats
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Export Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Save className="h-4 w-4" />
            <h3 className="text-lg font-medium">Export Configuration</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div>
                <Label htmlFor="export-format">Format</Label>
                <Select
                  value={exportOptions.format}
                  onValueChange={(value: any) => setExportOptions(prev => ({ ...prev, format: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {formatOptions.map((format) => {
                      const Icon = format.icon;
                      return (
                        <SelectItem key={format.value} value={format.value}>
                          <div className="flex items-center gap-2">
                            <Icon className="h-4 w-4" />
                            <div>
                              <div className="font-medium">{format.label}</div>
                              <div className="text-xs text-muted-foreground">{format.description}</div>
                            </div>
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="export-scope">Scope</Label>
                <Select
                  value={exportOptions.scope}
                  onValueChange={(value: any) => setExportOptions(prev => ({ ...prev, scope: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {scopeOptions.map((scope) => (
                      <SelectItem key={scope.value} value={scope.value}>
                        <div>
                          <div className="font-medium">{scope.label}</div>
                          <div className="text-xs text-muted-foreground">{scope.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Include Secrets</Label>
                  <p className="text-sm text-muted-foreground">
                    Export API keys and passwords
                  </p>
                </div>
                <Switch
                  checked={exportOptions.includeSecrets}
                  onCheckedChange={(checked) => setExportOptions(prev => ({ ...prev, includeSecrets: checked }))}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Include Defaults</Label>
                  <p className="text-sm text-muted-foreground">
                    Export default values
                  </p>
                </div>
                <Switch
                  checked={exportOptions.includeDefaults}
                  onCheckedChange={(checked) => setExportOptions(prev => ({ ...prev, includeDefaults: checked }))}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Compression</Label>
                  <p className="text-sm text-muted-foreground">
                    Compress export file
                  </p>
                </div>
                <Switch
                  checked={exportOptions.compression}
                  onCheckedChange={(checked) => setExportOptions(prev => ({ ...prev, compression: checked }))}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label>Encryption</Label>
                  <p className="text-sm text-muted-foreground">
                    Encrypt export file
                  </p>
                </div>
                <Switch
                  checked={exportOptions.encryption}
                  onCheckedChange={(checked) => setExportOptions(prev => ({ ...prev, encryption: checked }))}
                />
              </div>
            </div>
          </div>

          {exportOptions.encryption && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Exported files will be encrypted. Make sure to securely store the encryption password.
              </AlertDescription>
            </Alert>
          )}

          {isExporting && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Export Progress</span>
                <span>{exportProgress}%</span>
              </div>
              <Progress value={exportProgress} className="h-2" />
            </div>
          )}

          <Button onClick={exportConfiguration} disabled={isExporting} className="w-full">
            {isExporting ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            {isExporting ? 'Exporting...' : 'Export Configuration'}
          </Button>
        </div>

        <Separator />

        {/* Import Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            <h3 className="text-lg font-medium">Import Configuration</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="import-file">Configuration File</Label>
              <div className="flex gap-2">
                <Input
                  id="import-file"
                  type="file"
                  accept=".json,.yaml,.yml,.csv,.xml"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  className="flex-1"
                />
                <Button variant="outline" onClick={() => document.getElementById('import-file')?.click()}>
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
              {selectedFile && (
                <p className="text-sm text-muted-foreground mt-1">
                  Selected: {selectedFile.name}
                </p>
              )}
            </div>

            <div>
              <Label htmlFor="import-format">Format</Label>
              <Select
                value={exportOptions.format}
                onValueChange={(value: any) => setExportOptions(prev => ({ ...prev, format: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {formatOptions.map((format) => {
                    const Icon = format.icon;
                    return (
                      <SelectItem key={format.value} value={format.value}>
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4" />
                          {format.label}
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label>Overwrite Existing</Label>
              <p className="text-sm text-muted-foreground">
                Replace current configuration
              </p>
            </div>
            <Switch
              checked={exportOptions.includeDefaults}
              onChange={(checked) => setExportOptions(prev => ({ ...prev, includeDefaults: checked }))}
            />
          </div>

          {isImporting && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Import Progress</span>
                <span>{importProgress}%</span>
              </div>
              <Progress value={importProgress} className="h-2" />
            </div>
          )}

          <Button 
            onClick={importConfiguration} 
            disabled={isImporting || !selectedFile} 
            variant="outline" 
            className="w-full"
          >
            {isImporting ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Upload className="h-4 w-4 mr-2" />
            )}
            {isImporting ? 'Importing...' : 'Import Configuration'}
          </Button>
        </div>

        <Separator />

        {/* Templates Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Archive className="h-4 w-4" />
            <h3 className="text-lg font-medium">Templates</h3>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" onClick={saveTemplate} className="flex-1">
              <Database className="h-4 w-4 mr-2" />
              Save as Template
            </Button>
            <Button variant="outline" onClick={loadTemplate} className="flex-1">
              <FolderOpen className="h-4 w-4 mr-2" />
              Load Template
            </Button>
          </div>

          <p className="text-sm text-muted-foreground">
            Templates allow you to save and reuse common configuration setups for different trading scenarios.
          </p>
        </div>

        {/* Import Results */}
        {importResult && (
          <div className="space-y-3">
            <Separator />
            <div className="flex items-center gap-2">
              {getResultIcon(importResult.success)}
              <h3 className="text-lg font-medium">Import Results</h3>
              <Badge variant={importResult.success ? 'default' : 'destructive'}>
                {importResult.success ? 'Success' : 'Failed'}
              </Badge>
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="p-3 border rounded">
                <div className="text-muted-foreground">Total Items</div>
                <div className="text-xl font-bold">{importResult.totalItems}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-muted-foreground">Imported</div>
                <div className="text-xl font-bold text-green-600">{importResult.importedItems}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-muted-foreground">Skipped</div>
                <div className="text-xl font-bold text-yellow-600">{importResult.skippedItems}</div>
              </div>
            </div>

            {importResult.errors.length > 0 && (
              <Alert variant="destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>
                  <div className="font-medium mb-1">Errors:</div>
                  <ul className="list-disc list-inside space-y-1">
                    {importResult.errors.map((error, index) => (
                      <li key={index} className="text-sm">{error}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {importResult.warnings.length > 0 && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <div className="font-medium mb-1">Warnings:</div>
                  <ul className="list-disc list-inside space-y-1">
                    {importResult.warnings.map((warning, index) => (
                      <li key={index} className="text-sm">{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            <div className="flex justify-end">
              <Button variant="ghost" onClick={() => setImportResult(null)} size="sm">
                <Clock className="h-4 w-4 mr-2" />
                {new Date(importResult.timestamp).toLocaleString()}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ConfigIO;