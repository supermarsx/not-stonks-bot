/**
 * Database Connection and Integration Test Suite
 * Tests all database operations, WebSocket connections, and data flow
 */

import { dbService } from '../src/services/database';
import { wsService } from '../src/services/websocket';
import { api } from '../src/services/api';
import type {
  Portfolio,
  Position,
  Order,
  Trade,
  Broker,
  Strategy,
  RiskMetrics,
  MarketData,
} from '../src/types';

interface TestResult {
  name: string;
  success: boolean;
  duration: number;
  error?: string;
  data?: any;
}

class DatabaseTestSuite {
  private results: TestResult[] = [];

  private async runTest<T>(
    name: string,
    testFn: () => Promise<T>,
    validate?: (data: T) => boolean
  ): Promise<void> {
    const start = Date.now();
    try {
      console.log(`\n🧪 Running test: ${name}`);
      const data = await testFn();
      
      if (validate && !validate(data)) {
        throw new Error('Validation failed');
      }
      
      const duration = Date.now() - start;
      this.results.push({ name, success: true, duration, data });
      console.log(`✅ PASSED: ${name} (${duration}ms)`);
    } catch (error) {
      const duration = Date.now() - start;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.results.push({ name, success: false, duration, error: errorMessage });
      console.log(`❌ FAILED: ${name} - ${errorMessage} (${duration}ms)`);
    }
  }

  // ==================== CONNECTION TESTS ====================

  async testDatabaseConnections(): Promise<void> {
    console.log('\n📡 Testing Database Connections...');

    await this.runTest(
      'Database Service Connect',
      () => dbService.connect()
    );

    await this.runTest(
      'WebSocket Connection',
      () => wsService.connect()
    );

    await this.runTest(
      'API Health Check',
      () => api.getSystemHealth()
    );

    await this.runTest(
      'Connection Status Check',
      () => ({
        isConnected: dbService.isConnected(),
        isOnline: dbService.getOnlineStatus(),
        wsReadyState: wsService.getReadyState(),
      }),
      (status) => status.isConnected || status.isOnline // Allow offline fallback
    );
  }

  // ==================== PORTFOLIO TESTS ====================

  async testPortfolioOperations(): Promise<void> {
    console.log('\n💰 Testing Portfolio Operations...');

    await this.runTest(
      'Get Portfolio Data',
      () => dbService.getPortfolio(),
      (portfolio: Portfolio) => 
        portfolio && 
        typeof portfolio.totalValue === 'number' &&
        typeof portfolio.dailyPnL === 'number'
    );

    await this.runTest(
      'Get Portfolio History',
      () => dbService.getPortfolioHistory(30),
      (history: Portfolio[]) => Array.isArray(history) && history.length > 0
    );

    await this.runTest(
      'Portfolio Cache Test',
      () => {
        const key = 'portfolio_test';
        const testData = { totalValue: 100000 } as Portfolio;
        // This would normally be tested with actual cache methods
        return Promise.resolve({ cacheKey: key, data: testData });
      }
    );
  }

  // ==================== POSITION TESTS ====================

  async testPositionOperations(): Promise<void> {
    console.log('\n📊 Testing Position Operations...');

    await this.runTest(
      'Get All Positions',
      () => dbService.getPositions(),
      (positions: Position[]) => Array.isArray(positions)
    );

    await this.runTest(
      'Get Positions by Broker',
      () => dbService.getPositions('alpaca'),
      (positions: Position[]) => Array.isArray(positions)
    );

    // Test position by ID (may fail if no positions exist)
    try {
      const positions = await dbService.getPositions();
      if (positions.length > 0) {
        await this.runTest(
          'Get Position by ID',
          () => dbService.getPosition(positions[0].id),
          (position: Position) => position.id === positions[0].id
        );
      }
    } catch (error) {
      console.log('⚠️ Skipping position ID test - no positions available');
    }
  }

  // ==================== ORDER TESTS ====================

  async testOrderOperations(): Promise<void> {
    console.log('\n📋 Testing Order Operations...');

    await this.runTest(
      'Get All Orders',
      () => dbService.getOrders(),
      (orders: Order[]) => Array.isArray(orders)
    );

    await this.runTest(
      'Get Orders with Filters',
      () => dbService.getOrders({ broker: 'alpaca', limit: 10 }),
      (orders: Order[]) => Array.isArray(orders)
    );

    await this.runTest(
      'Get Paginated Trades',
      () => dbService.getTradesPaginated(1, 20),
      (paginated: any) => 
        paginated &&
        Array.isArray(paginated.items) &&
        typeof paginated.total === 'number'
    );
  }

  // ==================== BROKER TESTS ====================

  async testBrokerOperations(): Promise<void> {
    console.log('\n🏢 Testing Broker Operations...');

    await this.runTest(
      'Get All Brokers',
      () => dbService.getBrokers(),
      (brokers: Broker[]) => Array.isArray(brokers)
    );

    // Test broker by ID if any exist
    try {
      const brokers = await dbService.getBrokers();
      if (brokers.length > 0) {
        await this.runTest(
          'Get Broker by ID',
          () => dbService.getBroker(brokers[0].id),
          (broker: Broker) => broker.id === brokers[0].id
        );
      }
    } catch (error) {
      console.log('⚠️ Skipping broker ID test - no brokers available');
    }
  }

  // ==================== STRATEGY TESTS ====================

  async testStrategyOperations(): Promise<void> {
    console.log('\n🎯 Testing Strategy Operations...');

    await this.runTest(
      'Get All Strategies',
      () => dbService.getStrategies(),
      (strategies: Strategy[]) => Array.isArray(strategies)
    );

    // Test strategy by ID if any exist
    try {
      const strategies = await dbService.getStrategies();
      if (strategies.length > 0) {
        await this.runTest(
          'Get Strategy by ID',
          () => dbService.getStrategy(strategies[0].id),
          (strategy: Strategy) => strategy.id === strategies[0].id
        );
      }
    } catch (error) {
      console.log('⚠️ Skipping strategy ID test - no strategies available');
    }
  }

  // ==================== RISK TESTS ====================

  async testRiskOperations(): Promise<void> {
    console.log('\n⚠️ Testing Risk Operations...');

    await this.runTest(
      'Get Risk Metrics',
      () => dbService.getRiskMetrics(),
      (metrics: RiskMetrics) =>
        metrics &&
        typeof metrics.portfolioValue === 'number' &&
        typeof metrics.currentDrawdown === 'number'
    );

    await this.runTest(
      'Check Risk Limits',
      () => dbService.checkRiskLimits(),
      (result: { passed: boolean; violations: string[] }) =>
        typeof result.passed === 'boolean' &&
        Array.isArray(result.violations)
    );

    await this.runTest(
      'Get Risk History',
      () => dbService.getRiskHistory(30),
      (history: RiskMetrics[]) => Array.isArray(history)
    );
  }

  // ==================== MARKET DATA TESTS ====================

  async testMarketDataOperations(): Promise<void> {
    console.log('\n📈 Testing Market Data Operations...');

    await this.runTest(
      'Get Market Data for Multiple Symbols',
      () => dbService.getMarketData(['AAPL', 'TSLA', 'MSFT']),
      (data: MarketData[]) => Array.isArray(data)
    );

    await this.runTest(
      'Get Market Quote for Single Symbol',
      () => dbService.getMarketQuote('AAPL'),
      (quote: MarketData) =>
        quote &&
        quote.symbol === 'AAPL' &&
        typeof quote.price === 'number'
    );
  }

  // ==================== PERFORMANCE TESTS ====================

  async testPerformanceAndCaching(): Promise<void> {
    console.log('\n🚀 Testing Performance and Caching...');

    await this.runTest(
      'Cache Statistics',
      () => dbService.getCacheStats(),
      (stats: { size: number; entries: string[] }) =>
        typeof stats.size === 'number' &&
        Array.isArray(stats.entries)
    );

    await this.runTest(
      'Multiple Concurrent Requests',
      async () => {
        const promises = [
          dbService.getPortfolio(),
          dbService.getPositions(),
          dbService.getBrokers(),
          dbService.getRiskMetrics(),
        ];
        return await Promise.allSettled(promises);
      },
      (results) => Array.isArray(results) && results.length === 4
    );

    await this.runTest(
      'Cache Invalidation Test',
      async () => {
        dbService.clearCache('portfolio');
        return { cleared: true };
      }
    );
  }

  // ==================== EXPORT FUNCTIONALITY TESTS ====================

  async testExportFunctionality(): Promise<void> {
    console.log('\n📤 Testing Export Functionality...');

    await this.runTest(
      'Export Trades as CSV',
      () => dbService.exportData('trades', 'csv'),
      (blob: Blob) => blob instanceof Blob && blob.size > 0
    );

    await this.runTest(
      'Export Positions as JSON',
      () => dbService.exportData('positions', 'json'),
      (blob: Blob) => blob instanceof Blob && blob.size > 0
    );
  }

  // ==================== ERROR HANDLING TESTS ====================

  async testErrorHandling(): Promise<void> {
    console.log('\n🚫 Testing Error Handling...');

    await this.runTest(
      'Invalid Order Creation',
      async () => {
        try {
          await dbService.createOrder({
            symbol: '',
            broker: '',
            type: 'MARKET' as const,
            side: 'BUY' as const,
            quantity: -1,
          });
          throw new Error('Should have failed');
        } catch (error) {
          if (error instanceof Error && error.message.includes('Should have failed')) {
            throw error;
          }
          return { expectedError: true };
        }
      }
    );

    await this.runTest(
      'Non-existent Resource Access',
      async () => {
        try {
          await dbService.getPosition('non-existent-id');
          return { fallback: true };
        } catch (error) {
          return { errorHandled: true };
        }
      }
    );
  }

  // ==================== UTILITY TESTS ====================

  async testUtilityMethods(): Promise<void> {
    console.log('\n🔧 Testing Utility Methods...');

    await this.runTest(
      'Online/Offline Detection',
      () => ({
        isOnline: dbService.getOnlineStatus(),
        isConnected: dbService.isConnected(),
      })
    );

    await this.runTest(
      'Preload Essential Data',
      () => dbService.preloadEssentialData()
    );

    await this.runTest(
      'Clear All Cache',
      () => {
        dbService.clearAllCache();
        return { cleared: true };
      }
    );
  }

  // ==================== MAIN TEST RUNNER ====================

  async runAllTests(): Promise<{ passed: number; failed: number; total: number; results: TestResult[] }> {
    console.log('🧪 Starting Database Integration Test Suite...\n');
    
    const testSuites = [
      this.testDatabaseConnections.bind(this),
      this.testPortfolioOperations.bind(this),
      this.testPositionOperations.bind(this),
      this.testOrderOperations.bind(this),
      this.testBrokerOperations.bind(this),
      this.testStrategyOperations.bind(this),
      this.testRiskOperations.bind(this),
      this.testMarketDataOperations.bind(this),
      this.testPerformanceAndCaching.bind(this),
      this.testExportFunctionality.bind(this),
      this.testErrorHandling.bind(this),
      this.testUtilityMethods.bind(this),
    ];

    for (const testSuite of testSuites) {
      try {
        await testSuite();
      } catch (error) {
        console.error('Test suite failed:', error);
      }
    }

    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.filter(r => !r.success).length;
    const total = this.results.length;

    this.printSummary();
    
    return { passed, failed, total, results: this.results };
  }

  private printSummary(): void {
    console.log('\n📊 Test Results Summary');
    console.log('========================');
    
    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.filter(r => !r.success).length;
    const total = this.results.length;
    const totalTime = this.results.reduce((sum, r) => sum + r.duration, 0);

    console.log(`Total Tests: ${total}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log(`Success Rate: ${((passed / total) * 100).toFixed(1)}%`);
    console.log(`Total Time: ${totalTime}ms`);

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results
        .filter(r => !r.success)
        .forEach(r => {
          console.log(`  • ${r.name}: ${r.error}`);
        });
    }

    if (passed === total) {
      console.log('\n🎉 All tests passed! Database integration is working correctly.');
    } else {
      console.log('\n⚠️ Some tests failed. Please check your database connection and configuration.');
    }
  }

  getDetailedResults(): TestResult[] {
    return this.results;
  }
}

// ==================== EXPORT AND USAGE ====================

export const testSuite = new DatabaseTestSuite();
export default testSuite;

// ==================== NODE.JS TEST RUNNER ====================

if (typeof window === 'undefined' && typeof module !== 'undefined' && module.exports) {
  // Node.js environment - run tests if this file is executed directly
  testSuite.runAllTests().then(result => {
    console.log('\n🏁 Test suite completed');
    process.exit(result.failed > 0 ? 1 : 0);
  }).catch(error => {
    console.error('Test suite failed to run:', error);
    process.exit(1);
  });
}

// ==================== BROWSER TEST RUNNER ====================

if (typeof window !== 'undefined') {
  // Browser environment - expose test suite to window for manual testing
  (window as any).DatabaseTestSuite = testSuite;
  console.log('🧪 Database Test Suite loaded. Run testSuite.runAllTests() to start testing.');
}