import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MatrixCard } from '@/components/ui/MatrixCard';
import { MatrixButton } from '@/components/ui/MatrixButton';
import { MatrixInput } from '@/components/ui/MatrixInput';
import { 
  PieChart, 
  DollarSign, 
  TrendingUp, 
  Settings,
  Plus,
  Trash2,
  Save,
  RotateCcw,
  Target,
  BarChart3,
  Shield,
  AlertCircle,
  CheckCircle,
  Info
} from 'lucide-react';

interface AssetAllocation {
  id: string;
  symbol: string;
  name: string;
  category: 'equity' | 'bond' | 'commodity' | 'crypto' | 'forex' | 'etf' | 'fund';
  targetAllocation: number; // 0-100%
  currentAllocation: number; // 0-100%
  riskLevel: 'low' | 'medium' | 'high';
  volatility: number;
  expectedReturn: number;
  minInvestment?: number;
  maxAllocation?: number;
  rebalanceFrequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
}

interface SectorAllocation {
  id: string;
  name: string;
  targetAllocation: number;
  currentAllocation: number;
  assets: string[]; // Asset IDs
}

interface RegionAllocation {
  id: string;
  name: string;
  country: string;
  targetAllocation: number;
  currentAllocation: number;
  currency: string;
}

interface AllocationStrategy {
  id: string;
  name: string;
  description: string;
  totalValue: number;
  assets: AssetAllocation[];
  sectors: SectorAllocation[];
  regions: RegionAllocation[];
  rebalanceStrategy: {
    frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'when_drift';
    threshold: number; // Percentage drift to trigger rebalance
    minTrade: number; // Minimum trade size
    fees: number; // Trading fees percentage
  };
  constraints: {
    maxPositionSize: number;
    maxSectorExposure: number;
    maxRegionExposure: number;
    minCashReserve: number;
  };
  riskMetrics: {
    targetVolatility: number;
    maxDrawdown: number;
    sharpeRatio: number;
  };
}

interface AllocationBuilderProps {
  strategy: AllocationStrategy;
  onChange: (strategy: AllocationStrategy) => void;
  onSave?: () => void;
}

export const AllocationBuilder: React.FC<AllocationBuilderProps> = ({
  strategy,
  onChange,
  onSave
}) => {
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'assets' | 'sectors' | 'regions' | 'rebalance' | 'constraints'>('assets');
  const [showAddAsset, setShowAddAsset] = useState(false);

  const assetCategories = {
    equity: { name: 'Equities', color: 'border-blue-500', icon: <TrendingUp className="w-4 h-4" /> },
    bond: { name: 'Bonds', color: 'border-green-500', icon: <Shield className="w-4 h-4" /> },
    commodity: { name: 'Commodities', color: 'border-orange-500', icon: <Target className="w-4 h-4" /> },
    crypto: { name: 'Cryptocurrency', color: 'border-purple-500', icon: <BarChart3 className="w-4 h-4" /> },
    forex: { name: 'Forex', color: 'border-yellow-500', icon: <DollarSign className="w-4 h-4" /> },
    etf: { name: 'ETFs', color: 'border-cyan-500', icon: <PieChart className="w-4 h-4" /> },
    fund: { name: 'Funds', color: 'border-pink-500', icon: <PieChart className="w-4 h-4" /> }
  };

  const defaultAssets: Omit<AssetAllocation, 'id'>[] = [
    { symbol: 'AAPL', name: 'Apple Inc.', category: 'equity', targetAllocation: 5, currentAllocation: 4.8, riskLevel: 'medium', volatility: 0.25, expectedReturn: 0.12 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', category: 'equity', targetAllocation: 4, currentAllocation: 4.2, riskLevel: 'medium', volatility: 0.28, expectedReturn: 0.14 },
    { symbol: 'TSLA', name: 'Tesla Inc.', category: 'equity', targetAllocation: 3, currentAllocation: 3.1, riskLevel: 'high', volatility: 0.45, expectedReturn: 0.18 },
    { symbol: 'BTC', name: 'Bitcoin', category: 'crypto', targetAllocation: 2, currentAllocation: 1.9, riskLevel: 'high', volatility: 0.65, expectedReturn: 0.25 },
    { symbol: 'GLD', name: 'Gold ETF', category: 'etf', targetAllocation: 8, currentAllocation: 8.2, riskLevel: 'low', volatility: 0.15, expectedReturn: 0.08 },
    { symbol: 'VTI', name: 'Total Stock Market ETF', category: 'etf', targetAllocation: 15, currentAllocation: 14.8, riskLevel: 'medium', volatility: 0.18, expectedReturn: 0.10 },
    { symbol: 'VXUS', name: 'International ETF', category: 'etf', targetAllocation: 10, currentAllocation: 10.1, riskLevel: 'medium', volatility: 0.20, expectedReturn: 0.09 },
    { symbol: 'BND', name: 'Total Bond Market ETF', category: 'etf', targetAllocation: 20, currentAllocation: 19.9, riskLevel: 'low', volatility: 0.06, expectedReturn: 0.04 },
  ];

  const sectorAllocations: SectorAllocation[] = [
    { id: 'tech', name: 'Technology', targetAllocation: 25, currentAllocation: 24.1, assets: ['AAPL', 'GOOGL', 'TSLA'] },
    { id: 'healthcare', name: 'Healthcare', targetAllocation: 15, currentAllocation: 15.3, assets: ['JNJ', 'PFE', 'UNH'] },
    { id: 'financial', name: 'Financial', targetAllocation: 12, currentAllocation: 11.8, assets: ['JPM', 'BAC', 'WFC'] },
    { id: 'energy', name: 'Energy', targetAllocation: 8, currentAllocation: 8.2, assets: ['XOM', 'CVX', 'COP'] },
    { id: 'consumer', name: 'Consumer', targetAllocation: 10, currentAllocation: 10.1, assets: ['WMT', 'PG', 'KO'] },
    { id: 'industrial', name: 'Industrial', targetAllocation: 8, currentAllocation: 7.9, assets: ['BA', 'CAT', 'MMM'] },
  ];

  const regionAllocations: RegionAllocation[] = [
    { id: 'usa', name: 'United States', country: 'US', targetAllocation: 60, currentAllocation: 59.2, currency: 'USD' },
    { id: 'europe', name: 'Europe', country: 'EU', targetAllocation: 20, currentAllocation: 20.8, currency: 'EUR' },
    { id: 'asia', name: 'Asia Pacific', country: 'ASIA', targetAllocation: 15, currentAllocation: 15.1, currency: 'USD' },
    { id: 'emerging', name: 'Emerging Markets', country: 'EM', targetAllocation: 5, currentAllocation: 4.9, currency: 'USD' },
  ];

  const addAsset = () => {
    const newAsset: AssetAllocation = {
      id: `asset_${Date.now()}`,
      symbol: 'NEW',
      name: 'New Asset',
      category: 'equity',
      targetAllocation: 0,
      currentAllocation: 0,
      riskLevel: 'medium',
      volatility: 0.2,
      expectedReturn: 0.1,
      rebalanceFrequency: 'monthly'
    };

    onChange({
      ...strategy,
      assets: [...strategy.assets, newAsset]
    });
    setSelectedAsset(newAsset.id);
    setShowAddAsset(false);
  };

  const updateAsset = (assetId: string, updates: Partial<AssetAllocation>) => {
    onChange({
      ...strategy,
      assets: strategy.assets.map(asset =>
        asset.id === assetId ? { ...asset, ...updates } : asset
      )
    });
  };

  const removeAsset = (assetId: string) => {
    onChange({
      ...strategy,
      assets: strategy.assets.filter(asset => asset.id !== assetId)
    });
    if (selectedAsset === assetId) {
      setSelectedAsset(null);
    }
  };

  const calculateTotalAllocation = () => {
    return strategy.assets.reduce((sum, asset) => sum + asset.targetAllocation, 0);
  };

  const calculateDrift = (asset: AssetAllocation) => {
    return asset.currentAllocation - asset.targetAllocation;
  };

  const needsRebalance = (asset: AssetAllocation) => {
    const drift = Math.abs(calculateDrift(asset));
    return drift > strategy.rebalanceStrategy.threshold;
  };

  const getCategoryInfo = (category: AssetAllocation['category']) => {
    return assetCategories[category];
  };

  const renderAssetConfig = (asset: AssetAllocation) => {
    const categoryInfo = getCategoryInfo(asset.category);
    const drift = calculateDrift(asset);
    const needsRebalanceFlag = needsRebalance(asset);

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Symbol</label>
            <MatrixInput
              value={asset.symbol}
              onChange={(e) => updateAsset(asset.id, { symbol: e.target.value })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Name</label>
            <MatrixInput
              value={asset.name}
              onChange={(e) => updateAsset(asset.id, { name: e.target.value })}
              className="text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">Category</label>
            <select
              value={asset.category}
              onChange={(e) => updateAsset(asset.id, { category: e.target.value as AssetAllocation['category'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              {Object.entries(assetCategories).map(([key, info]) => (
                <option key={key} value={key}>
                  {info.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">Risk Level</label>
            <select
              value={asset.riskLevel}
              onChange={(e) => updateAsset(asset.id, { riskLevel: e.target.value as AssetAllocation['riskLevel'] })}
              className="matrix-input w-full px-3 py-2 text-sm"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Target Allocation (%)
            </label>
            <div className="relative">
              <MatrixInput
                type="number"
                min="0"
                max="100"
                step="0.1"
                value={asset.targetAllocation}
                onChange={(e) => updateAsset(asset.id, { targetAllocation: parseFloat(e.target.value) })}
                className="text-sm"
              />
              <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
            </div>
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Current Allocation (%)
            </label>
            <div className="relative">
              <MatrixInput
                type="number"
                min="0"
                max="100"
                step="0.1"
                value={asset.currentAllocation}
                onChange={(e) => updateAsset(asset.id, { currentAllocation: parseFloat(e.target.value) })}
                className="text-sm"
              />
              <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Expected Return (%)
            </label>
            <div className="relative">
              <MatrixInput
                type="number"
                step="0.1"
                value={asset.expectedReturn * 100}
                onChange={(e) => updateAsset(asset.id, { expectedReturn: parseFloat(e.target.value) / 100 })}
                className="text-sm"
              />
              <span className="absolute right-3 top-2 text-xs text-green-600">%</span>
            </div>
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Volatility (σ)
            </label>
            <MatrixInput
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={asset.volatility}
              onChange={(e) => updateAsset(asset.id, { volatility: parseFloat(e.target.value) })}
              className="text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Min Investment ($)
            </label>
            <MatrixInput
              type="number"
              min="0"
              value={asset.minInvestment || 0}
              onChange={(e) => updateAsset(asset.id, { minInvestment: parseFloat(e.target.value) })}
              className="text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-green-400 mb-1">
              Max Allocation (%)
            </label>
            <MatrixInput
              type="number"
              min="0"
              max="100"
              value={asset.maxAllocation || 100}
              onChange={(e) => updateAsset(asset.id, { maxAllocation: parseFloat(e.target.value) })}
              className="text-sm"
            />
          </div>
        </div>

        {/* Drift and Rebalance Status */}
        <div className="matrix-card p-3 bg-black/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-green-400">Portfolio Drift</span>
            <div className="flex items-center gap-2">
              {needsRebalanceFlag ? (
                <AlertCircle className="w-4 h-4 text-yellow-400" />
              ) : drift === 0 ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : (
                <Info className="w-4 h-4 text-blue-400" />
              )}
              <span className={`text-xs font-bold ${
                Math.abs(drift) > strategy.rebalanceStrategy.threshold ? 'text-yellow-400' :
                Math.abs(drift) === 0 ? 'text-green-400' : 'text-blue-400'
              }`}>
                {drift > 0 ? '+' : ''}{drift.toFixed(2)}%
              </span>
            </div>
          </div>
          
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all ${
                Math.abs(drift) > strategy.rebalanceStrategy.threshold ? 'bg-yellow-500' :
                Math.abs(drift) === 0 ? 'bg-green-500' : 'bg-blue-500'
              }`}
              style={{ width: `${Math.min(100, Math.abs(drift) * 20)}%` }}
            />
          </div>
          
          {needsRebalanceFlag && (
            <div className="text-xs text-yellow-400 mt-1">
              Rebalancing needed (threshold: ±{strategy.rebalanceStrategy.threshold}%)
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-green-800/30">
        <div>
          <h1 className="text-2xl font-bold matrix-text-glow text-green-400">
            PORTFOLIO ALLOCATION BUILDER
          </h1>
          <p className="text-green-600 text-sm">Build and manage asset allocation strategies</p>
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Total Portfolio Value</div>
            <div className="text-lg font-bold text-green-400">
              ${strategy.totalValue.toLocaleString()}
            </div>
          </div>
          
          <div className="text-right mr-4">
            <div className="text-xs text-green-600">Total Allocation</div>
            <div className={`text-lg font-bold ${
              Math.abs(calculateTotalAllocation() - 100) < 0.1 ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {calculateTotalAllocation().toFixed(1)}%
            </div>
          </div>
          
          {onSave && (
            <MatrixButton onClick={onSave}>
              <Save className="w-4 h-4 mr-2" />
              Save Allocation
            </MatrixButton>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-green-800/30">
        {[
          { id: 'assets', label: 'Asset Allocation', icon: <DollarSign className="w-4 h-4" /> },
          { id: 'sectors', label: 'Sector Allocation', icon: <PieChart className="w-4 h-4" /> },
          { id: 'regions', label: 'Regional Allocation', icon: <BarChart3 className="w-4 h-4" /> },
          { id: 'rebalance', label: 'Rebalancing', icon: <RotateCcw className="w-4 h-4" /> },
          { id: 'constraints', label: 'Constraints', icon: <Shield className="w-4 h-4" /> }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id 
                ? 'text-green-400 border-b-2 border-green-500' 
                : 'text-green-600 hover:text-green-400'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Asset List */}
        <div className="w-80 border-r border-green-800/30 flex flex-col">
          <div className="p-4 border-b border-green-800/30">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-green-400">ASSETS</h3>
              <MatrixButton 
                size="sm" 
                onClick={addAsset}
                className="flex items-center gap-1"
              >
                <Plus className="w-3 h-3" />
                Add
              </MatrixButton>
            </div>
            
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {strategy.assets.map((asset) => {
                const categoryInfo = getCategoryInfo(asset.category);
                const drift = calculateDrift(asset);
                const needsRebalanceFlag = needsRebalance(asset);
                
                return (
                  <MatrixCard
                    key={asset.id}
                    className={`p-3 cursor-pointer transition-all ${
                      selectedAsset === asset.id 
                        ? 'border-green-500 bg-green-900/20' 
                        : 'border-green-800/30 hover:border-green-700'
                    }`}
                    onClick={() => setSelectedAsset(asset.id)}
                  >
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={categoryInfo.color}>
                            {categoryInfo.icon}
                          </div>
                          <div>
                            <div className="text-sm font-bold text-green-400">
                              {asset.symbol}
                            </div>
                            <div className="text-xs text-green-600 truncate">
                              {asset.name}
                            </div>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className="text-xs font-bold text-green-400">
                            {asset.targetAllocation}%
                          </div>
                          <div className={`text-xs ${
                            Math.abs(drift) > strategy.rebalanceStrategy.threshold ? 'text-yellow-400' : 'text-green-600'
                          }`}>
                            {drift > 0 ? '+' : ''}{drift.toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="text-xs text-green-600">
                          {categoryInfo.name}
                        </div>
                        
                        <div className="flex items-center gap-1">
                          {needsRebalanceFlag ? (
                            <AlertCircle className="w-3 h-3 text-yellow-400" />
                          ) : drift === 0 ? (
                            <CheckCircle className="w-3 h-3 text-green-400" />
                          ) : null}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              removeAsset(asset.id);
                            }}
                            className="text-red-400 hover:text-red-300"
                          >
                            <Trash2 className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </MatrixCard>
                );
              })}
            </div>
          </div>

          {/* Quick Add Assets */}
          <div className="flex-1 overflow-y-auto p-4">
            <h3 className="text-sm font-bold text-green-400 mb-3">QUICK ADD</h3>
            <div className="space-y-2">
              {defaultAssets.slice(0, 5).map((defaultAsset, index) => {
                const categoryInfo = getCategoryInfo(defaultAsset.category);
                return (
                  <MatrixCard
                    key={index}
                    className="p-2 cursor-pointer hover:bg-green-900/20 transition-colors"
                    onClick={() => {
                      const newAsset: AssetAllocation = {
                        ...defaultAsset,
                        id: `asset_${Date.now()}_${index}`,
                        currentAllocation: 0,
                        rebalanceFrequency: 'monthly'
                      };
                      onChange({
                        ...strategy,
                        assets: [...strategy.assets, newAsset]
                      });
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <div className={categoryInfo.color}>
                        {categoryInfo.icon}
                      </div>
                      <div className="flex-1">
                        <div className="text-xs font-bold text-green-400">
                          {defaultAsset.symbol}
                        </div>
                        <div className="text-xs text-green-600 truncate">
                          {defaultAsset.name}
                        </div>
                      </div>
                    </div>
                  </MatrixCard>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Panel - Configuration */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            {selectedAsset ? (
              <motion.div
                key="config"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                {(() => {
                  const asset = strategy.assets.find(a => a.id === selectedAsset);
                  if (!asset) return null;
                  
                  const categoryInfo = getCategoryInfo(asset.category);
                  return (
                    <MatrixCard className="p-6">
                      <div className="space-y-4">
                        <div className="flex items-center gap-3 mb-6">
                          <div className={`${categoryInfo.color} p-2 rounded`}>
                            {categoryInfo.icon}
                          </div>
                          <div>
                            <h2 className="text-lg font-bold text-green-400">
                              {asset.symbol} - {asset.name}
                            </h2>
                            <p className="text-sm text-green-600">
                              {categoryInfo.name} Allocation
                            </p>
                          </div>
                        </div>

                        {renderAssetConfig(asset)}
                      </div>
                    </MatrixCard>
                  );
                })()}
              </motion.div>
            ) : (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-6"
              >
                <div className="space-y-6">
                  {/* Portfolio Overview */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Portfolio Overview
                    </h3>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <div className="text-xs text-green-600">Total Assets</div>
                        <div className="text-2xl font-bold text-green-400">
                          {strategy.assets.length}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Target Return</div>
                        <div className="text-2xl font-bold text-green-400">
                          {(strategy.assets.reduce((sum, asset) => 
                            sum + (asset.expectedReturn * asset.targetAllocation / 100), 0
                          ) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Portfolio Volatility</div>
                        <div className="text-2xl font-bold text-green-400">
                          {(Math.sqrt(strategy.assets.reduce((sum, asset) => 
                            sum + Math.pow(asset.volatility * asset.targetAllocation / 100, 2), 0
                          )) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-green-600">Rebalance Threshold</div>
                        <div className="text-2xl font-bold text-yellow-400">
                          ±{strategy.rebalanceStrategy.threshold}%
                        </div>
                      </div>
                    </div>
                  </MatrixCard>

                  {/* Category Breakdown */}
                  <MatrixCard className="p-6">
                    <h3 className="text-lg font-bold text-green-400 mb-4">
                      Category Breakdown
                    </h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(assetCategories).map(([category, info]) => {
                        const categoryAssets = strategy.assets.filter(a => a.category === category);
                        const totalAllocation = categoryAssets.reduce((sum, asset) => sum + asset.targetAllocation, 0);
                        
                        return (
                          <div key={category} className="matrix-card p-4">
                            <div className="flex items-center gap-2 mb-2">
                              <div className={info.color}>
                                {info.icon}
                              </div>
                              <div className="text-sm font-bold text-green-400">
                                {info.name}
                              </div>
                            </div>
                            <div className="text-2xl font-bold text-green-400 mb-1">
                              {totalAllocation.toFixed(1)}%
                            </div>
                            <div className="text-xs text-green-600">
                              {categoryAssets.length} assets
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                              <div 
                                className="h-2 bg-green-500 rounded-full"
                                style={{ width: `${totalAllocation}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </MatrixCard>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};