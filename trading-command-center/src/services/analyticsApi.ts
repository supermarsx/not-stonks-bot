import axios from 'axios';

const ANALYTICS_API_BASE = 'http://localhost:8000/api/analytics';

const analyticsAxios = axios.create({
  baseURL: ANALYTICS_API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Performance Analytics API
export const analyticsApi = {
  // Performance endpoints
  getPerformanceDemoData: async () => {
    const response = await analyticsAxios.get('/performance/demo-data');
    return response.data;
  },

  calculatePerformanceMetrics: async (data: {
    returns: number[];
    dates: string[];
    risk_free_rate?: number;
    benchmark_returns?: number[];
  }) => {
    const response = await analyticsAxios.post('/performance/metrics', data);
    return response.data;
  },

  calculateRollingMetrics: async (data: {
    returns: number[];
    dates: string[];
    risk_free_rate?: number;
  }, windowDays: number = 252) => {
    const response = await analyticsAxios.post(`/performance/rolling-metrics?window_days=${windowDays}`, data);
    return response.data;
  },

  calculateAttribution: async (data: {
    portfolio_weights: Record<string, number>;
    benchmark_weights: Record<string, number>;
    portfolio_returns: Record<string, number>;
    benchmark_returns: Record<string, number>;
    portfolio_return: number;
    benchmark_return: number;
  }) => {
    const response = await analyticsAxios.post('/performance/attribution', data);
    return response.data;
  },

  calculateVarCvar: async (data: {
    returns: number[];
    dates: string[];
    risk_free_rate?: number;
  }) => {
    const response = await analyticsAxios.post('/performance/var-cvar', data);
    return response.data;
  },

  // Execution Analytics endpoints
  getDemoTrades: async () => {
    const response = await analyticsAxios.get('/execution/demo-trades');
    return response.data;
  },

  calculateImplementationShortfall: async (data: {
    decision_price: number;
    arrival_price: number;
    execution_price: number;
    trade_value: number;
    commission?: number;
    market_impact?: number;
  }) => {
    const response = await analyticsAxios.post('/execution/implementation-shortfall', data);
    return response.data;
  },

  analyzeVwapPerformance: async (data: {
    execution_price: number;
    vwap_price: number;
    trade_volume: number;
    total_volume: number;
    side?: string;
  }) => {
    const response = await analyticsAxios.post('/execution/vwap-analysis', data);
    return response.data;
  },

  calculateMarketImpact: async (data: {
    pre_trade_price: number;
    post_trade_price: number;
    trade_size: number;
    average_daily_volume: number;
    trade_duration_minutes?: number;
  }) => {
    const response = await analyticsAxios.post('/execution/market-impact', data);
    return response.data;
  },

  generateExecutionScorecard: async (trades: any[]) => {
    const response = await analyticsAxios.post('/execution/execution-scorecard', { trades });
    return response.data;
  },

  // Risk Analytics endpoints
  calculateMonteCarloVar: async (data: {
    returns: number[];
    confidence_levels?: number[];
    num_simulations?: number;
    time_horizon?: number;
  }) => {
    const response = await analyticsAxios.post('/risk/monte-carlo-var', data);
    return response.data;
  },

  calculateParametricVar: async (data: {
    portfolio_value: number;
    expected_return: number;
    volatility: number;
    confidence_levels?: number[];
    time_horizon?: number;
  }) => {
    const response = await analyticsAxios.post('/risk/parametric-var', data);
    return response.data;
  },

  performStressTesting: async (data: {
    portfolio_weights: Record<string, number>;
    asset_returns: Record<string, number[]>;
    stress_scenarios: Record<string, Record<string, number>>;
  }) => {
    const response = await analyticsAxios.post('/risk/stress-testing', data);
    return response.data;
  },

  analyzeCorrelation: async (data: {
    returns_data: Record<string, number[]>;
    dates?: string[];
  }) => {
    const response = await analyticsAxios.post('/risk/correlation-analysis', data);
    return response.data;
  },

  analyzeTailRisk: async (data: {
    returns: number[];
    tail_percentile?: number;
  }) => {
    const response = await analyticsAxios.post('/risk/tail-risk-analysis', data);
    return response.data;
  },

  calculateConcentrationRisk: async (data: {
    portfolio_weights: Record<string, number>;
    asset_returns?: Record<string, number[]>;
  }) => {
    const response = await analyticsAxios.post('/risk/concentration-risk', data);
    return response.data;
  },

  getDefaultStressScenarios: async () => {
    const response = await analyticsAxios.get('/risk/default-stress-scenarios');
    return response.data;
  },

  // Portfolio Optimization endpoints
  optimizeMeanVariance: async (data: {
    expected_returns: number[];
    covariance_matrix: number[][];
    asset_names: string[];
    risk_aversion?: number;
    weight_bounds?: [number, number];
    target_return?: number;
  }) => {
    const response = await analyticsAxios.post('/optimization/mean-variance', data);
    return response.data;
  },

  optimizeBlackLitterman: async (data: {
    market_caps: number[];
    returns_data: Record<string, number[]>;
    asset_names: string[];
    investor_views?: Record<string, number>;
    view_confidence?: number;
    risk_aversion?: number;
    tau?: number;
  }) => {
    const response = await analyticsAxios.post('/optimization/black-litterman', data);
    return response.data;
  },

  generateEfficientFrontier: async (data: {
    expected_returns: number[];
    covariance_matrix: number[][];
    asset_names: string[];
    num_portfolios?: number;
    weight_bounds?: [number, number];
  }) => {
    const response = await analyticsAxios.post('/optimization/efficient-frontier', data);
    return response.data;
  },

  optimizeRiskParity: async (data: {
    covariance_matrix: number[][];
    asset_names: string[];
    target_volatility?: number;
  }) => {
    const response = await analyticsAxios.post('/optimization/risk-parity', data);
    return response.data;
  },

  generateTacticalSignals: async (data: {
    current_weights: Record<string, number>;
    target_weights: Record<string, number>;
    returns_momentum: Record<string, number>;
    volatility_regime: Record<string, number>;
    rebalance_threshold?: number;
  }) => {
    const response = await analyticsAxios.post('/optimization/tactical-allocation', data);
    return response.data;
  },

  getDemoOptimizationData: async () => {
    const response = await analyticsAxios.get('/optimization/demo-optimization-data');
    return response.data;
  },

  // Reports endpoints
  generatePerformanceReport: async (data: {
    portfolio_name: string;
    returns: number[];
    dates: string[];
    benchmark_returns?: number[];
    benchmark_name?: string;
    period?: string;
    risk_free_rate?: number;
  }) => {
    const response = await analyticsAxios.post('/reports/performance', data);
    return response.data;
  },

  generateRiskReport: async (data: {
    portfolio_name: string;
    portfolio_weights: Record<string, number>;
    portfolio_returns: number[];
    asset_returns?: Record<string, number[]>;
    stress_scenarios?: Record<string, Record<string, number>>;
    portfolio_value?: number;
  }) => {
    const response = await analyticsAxios.post('/reports/risk', data);
    return response.data;
  },

  generateExecutionReport: async (data: {
    portfolio_name: string;
    trades: any[];
    period?: string;
    benchmark_type?: string;
  }) => {
    const response = await analyticsAxios.post('/reports/execution', data);
    return response.data;
  },

  generateOptimizationReport: async (data: {
    current_portfolio: Record<string, number>;
    optimization_results: any;
    optimization_method: string;
    expected_returns: number[];
    asset_names: string[];
  }) => {
    const response = await analyticsAxios.post('/reports/optimization', data);
    return response.data;
  },

  exportReport: async (data: {
    report_data: any;
    format: string;
    filename?: string;
  }) => {
    const response = await analyticsAxios.post('/reports/export', data);
    return response.data;
  },

  getReportTemplates: async () => {
    const response = await analyticsAxios.get('/reports/templates');
    return response.data;
  },

  // System endpoints
  getCapabilities: async () => {
    const response = await analyticsAxios.get('/capabilities');
    return response.data;
  },

  getHealth: async () => {
    const response = await analyticsAxios.get('/health', { baseURL: 'http://localhost:8000' });
    return response.data;
  },
};

export default analyticsApi;