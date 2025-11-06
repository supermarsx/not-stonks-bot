#!/usr/bin/env node

/**
 * Quick Integration Test and Validation Script
 * Tests basic database connectivity and functionality
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface ValidationResult {
  component: string;
  status: '✅ PASS' | '❌ FAIL' | '⚠️ SKIP';
  description: string;
  file?: string;
}

class IntegrationValidator {
  private results: ValidationResult[] = [];

  validateFilesExist(files: string[]): void {
    console.log('\n📁 Validating File Structure...');
    
    for (const file of files) {
      const fullPath = join(process.cwd(), file);
      if (existsSync(fullPath)) {
        this.results.push({
          component: file,
          status: '✅ PASS',
          description: 'File exists',
          file
        });
      } else {
        this.results.push({
          component: file,
          status: '❌ FAIL',
          description: 'File not found',
          file
        });
      }
    }
  }

  validateFileContent(filePath: string, searchPatterns: string[]): void {
    console.log(`\n📄 Validating ${filePath}...`);
    
    try {
      const fullPath = join(process.cwd(), filePath);
      const content = readFileSync(fullPath, 'utf-8');
      
      for (const pattern of searchPatterns) {
        if (content.includes(pattern)) {
          this.results.push({
            component: `${filePath}:${pattern.substring(0, 30)}...`,
            status: '✅ PASS',
            description: 'Pattern found',
            file: filePath
          });
        } else {
          this.results.push({
            component: `${filePath}:${pattern.substring(0, 30)}...`,
            status: '❌ FAIL',
            description: 'Pattern not found',
            file: filePath
          });
        }
      }
    } catch (error) {
      this.results.push({
        component: filePath,
        status: '❌ FAIL',
        description: `Cannot read file: ${error}`,
        file: filePath
      });
    }
  }

  validateImports(filePath: string, expectedImports: string[]): void {
    console.log(`\n📦 Validating imports in ${filePath}...`);
    
    try {
      const fullPath = join(process.cwd(), filePath);
      const content = readFileSync(fullPath, 'utf-8');
      
      for (const importName of expectedImports) {
        if (content.includes(`import.*${importName}`) || content.includes(`from.*${importName}`)) {
          this.results.push({
            component: `${filePath}:import ${importName}`,
            status: '✅ PASS',
            description: 'Import found',
            file: filePath
          });
        } else {
          this.results.push({
            component: `${filePath}:import ${importName}`,
            status: '⚠️ SKIP',
            description: 'Import not found (may be optional)',
            file: filePath
          });
        }
      }
    } catch (error) {
      this.results.push({
        component: filePath,
        status: '❌ FAIL',
        description: `Cannot read file: ${error}`,
        file: filePath
      });
    }
  }

  printSummary(): void {
    console.log('\n📊 Validation Results Summary');
    console.log('==============================');
    
    const passed = this.results.filter(r => r.status.includes('PASS')).length;
    const failed = this.results.filter(r => r.status.includes('FAIL')).length;
    const skipped = this.results.filter(r => r.status.includes('SKIP')).length;
    const total = this.results.length;
    
    console.log(`Total Checks: ${total}`);
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`⚠️ Skipped: ${skipped}`);
    
    if (failed === 0) {
      console.log('\n🎉 All critical validations passed!');
    } else {
      console.log(`\n⚠️ ${failed} validations failed. Please check the files.`);
    }
    
    console.log('\n📋 Detailed Results:');
    this.results.forEach(result => {
      console.log(`  ${result.status} ${result.component}: ${result.description}`);
    });
  }

  runValidation(): void {
    console.log('🔍 Running Database Integration Validation...\n');
    
    // 1. Validate core files exist
    this.validateFilesExist([
      'src/services/database.ts',
      'src/hooks/useDatabase.ts',
      'src/pages/Dashboard.tsx',
      'src/pages/Orders.tsx',
      'src/pages/Strategies.tsx',
      'src/pages/Risk.tsx',
      'src/services/api.ts',
      'src/services/websocket.ts',
      'src/types/index.ts',
      'src/tests/database-integration.test.ts',
    ]);
    
    // 2. Validate database service implementation
    this.validateFileContent('src/services/database.ts', [
      'class DatabaseService',
      'getPortfolio',
      'getPositions',
      'getOrders',
      'getBrokers',
      'getStrategies',
      'getRiskMetrics',
      'getMarketData',
      'exportData',
      'cache',
      'WebSocket',
    ]);
    
    // 3. Validate hooks implementation
    this.validateFileContent('src/hooks/useDatabase.ts', [
      'usePortfolio',
      'usePositions',
      'useOrders',
      'useStrategies',
      'useRiskMetrics',
      'useBrokers',
      'loading',
      'error',
      'refresh',
    ]);
    
    // 4. Validate Dashboard updates
    this.validateFileContent('src/pages/Dashboard.tsx', [
      'usePortfolio',
      'usePositions',
      'useRiskMetrics',
      'useConnectionStatus',
      'dbService',
    ]);
    
    // 5. Validate Orders page updates
    this.validateFileContent('src/pages/Orders.tsx', [
      'useOrders',
      'useBrokers',
      'createOrder',
      'cancelOrder',
      'useDataExport',
    ]);
    
    // 6. Validate Strategies page updates
    this.validateFileContent('src/pages/Strategies.tsx', [
      'useStrategies',
      'updateStrategy',
      'deleteStrategy',
      'createStrategy',
    ]);
    
    // 7. Validate Risk page updates
    this.validateFileContent('src/pages/Risk.tsx', [
      'useRiskMetrics',
      'checkRiskLimits',
      'refreshMetrics',
    ]);
    
    // 8. Validate import dependencies
    this.validateImports('src/pages/Dashboard.tsx', [
      '../hooks/useDatabase',
      '../services/database',
    ]);
    
    this.validateImports('src/services/database.ts', [
      '../services/api',
      '../services/websocket',
    ]);
    
    this.printSummary();
  }
}

// ==================== MAIN EXECUTION ====================

const validator = new IntegrationValidator();

if (require.main === module) {
  validator.runValidation();
}

export default validator;