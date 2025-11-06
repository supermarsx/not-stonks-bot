import React, { useState, useMemo } from 'react';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { BarChart3, TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface AssetCorrelation {
  asset1: string;
  asset2: string;
  correlation: number;
  pValue?: number;
  sampleSize?: number;
}

interface CorrelationMatrixProps {
  correlationData: AssetCorrelation[];
  title?: string;
  className?: string;
  assetNames?: { [key: string]: string };
  correlationThreshold?: number;
  interactive?: boolean;
  showValues?: boolean;
  onCorrelationClick?: (asset1: string, asset2: string, correlation: number) => void;
  onExport?: () => void;
}

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  correlationData,
  title = "Asset Correlation Matrix",
  className = "",
  assetNames = {},
  correlationThreshold = 0.7,
  interactive = true,
  showValues = true,
  onCorrelationClick,
  onExport,
}) => {
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [sortOrder, setSortOrder] = useState<'correlation' | 'asset'>('correlation');
  const [hoveredCell, setHoveredCell] = useState<{ asset1: string; asset2: string } | null>(null);

  // Extract unique assets
  const uniqueAssets = useMemo(() => {
    const assets = new Set<string>();
    correlationData.forEach(item => {
      assets.add(item.asset1);
      assets.add(item.asset2);
    });
    return Array.from(assets);
  }, [correlationData]);

  // Create correlation matrix
  const correlationMatrix = useMemo(() => {
    const matrix: { [key: string]: { [key: string]: number } } = {};
    
    // Initialize matrix
    uniqueAssets.forEach(asset1 => {
      matrix[asset1] = {};
      uniqueAssets.forEach(asset2 => {
        if (asset1 === asset2) {
          matrix[asset1][asset2] = 1;
        } else {
          matrix[asset1][asset2] = 0;
        }
      });
    });

    // Fill matrix with correlation data
    correlationData.forEach(item => {
      matrix[item.asset1][item.asset2] = item.correlation;
      matrix[item.asset2][item.asset1] = item.correlation; // Symmetric
    });

    return matrix;
  }, [correlationData, uniqueAssets]);

  // Get correlation color
  const getCorrelationColor = (correlation: number) => {
    if (correlation >= 0.8) return '#00ff00'; // Strong positive - Green
    if (correlation >= 0.5) return '#88ff00'; // Medium positive - Light green
    if (correlation >= 0.2) return '#ffff00'; // Weak positive - Yellow
    if (correlation >= -0.2) return '#666666'; // Neutral - Gray
    if (correlation >= -0.5) return '#ff8800'; // Weak negative - Orange
    if (correlation >= -0.8) return '#ff6600'; // Medium negative - Red-orange
    return '#ff0000'; // Strong negative - Red
  };

  // Get correlation intensity
  const getCorrelationIntensity = (correlation: number) => {
    return Math.abs(correlation);
  };

  // Get correlation strength label
  const getCorrelationStrength = (correlation: number) => {
    const abs = Math.abs(correlation);
    if (abs >= 0.8) return 'Strong';
    if (abs >= 0.5) return 'Medium';
    if (abs >= 0.2) return 'Weak';
    if (abs >= 0.05) return 'Very Weak';
    return 'None';
  };

  // Sort assets by correlation strength
  const sortedAssets = useMemo(() => {
    if (sortOrder === 'asset') {
      return [...uniqueAssets].sort();
    }
    
    // Sort by average correlation with others (descending)
    return [...uniqueAssets].sort((asset1, asset2) => {
      const avgCorr1 = uniqueAssets.reduce((sum, other) => {
        if (other === asset1) return sum;
        return sum + Math.abs(correlationMatrix[asset1][other]);
      }, 0) / (uniqueAssets.length - 1);
      
      const avgCorr2 = uniqueAssets.reduce((sum, other) => {
        if (other === asset2) return sum;
        return sum + Math.abs(correlationMatrix[asset2][other]);
      }, 0) / (uniqueAssets.length - 1);
      
      return avgCorr2 - avgCorr1;
    });
  }, [uniqueAssets, correlationMatrix, sortOrder]);

  // Calculate matrix statistics
  const matrixStats = useMemo(() => {
    const correlations: number[] = [];
    const positiveCorrelations: number[] = [];
    const negativeCorrelations: number[] = [];
    
    correlationData.forEach(item => {
      correlations.push(item.correlation);
      if (item.correlation > 0) {
        positiveCorrelations.push(item.correlation);
      } else {
        negativeCorrelations.push(item.correlation);
      }
    });
    
    const avgCorrelation = correlations.reduce((sum, corr) => sum + corr, 0) / correlations.length;
    const maxCorrelation = Math.max(...correlations);
    const minCorrelation = Math.min(...correlations);
    const highCorrelationPairs = correlationData.filter(item => Math.abs(item.correlation) >= correlationThreshold).length;
    
    return {
      avgCorrelation,
      maxCorrelation,
      minCorrelation,
      totalPairs: correlationData.length,
      highCorrelationPairs,
      avgPositive: positiveCorrelations.length > 0 ? positiveCorrelations.reduce((sum, corr) => sum + corr, 0) / positiveCorrelations.length : 0,
      avgNegative: negativeCorrelations.length > 0 ? negativeCorrelations.reduce((sum, corr) => sum + corr, 0) / negativeCorrelations.length : 0,
    };
  }, [correlationData, correlationThreshold]);

  // Handle cell click
  const handleCellClick = (asset1: string, asset2: string) => {
    if (!interactive) return;
    
    const correlation = correlationMatrix[asset1][asset2];
    onCorrelationClick?.(asset1, asset2, correlation);
  };

  // Get asset display name
  const getAssetName = (asset: string) => {
    return assetNames[asset] || asset;
  };

  // Filter matrix based on selected assets
  const filteredAssets = selectedAssets.length > 0 
    ? sortedAssets.filter(asset => selectedAssets.includes(asset))
    : sortedAssets;

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* Controls */}
        <div className="flex flex-wrap gap-2 items-center">
          <div className="flex gap-1">
            <MatrixButton
              size="sm"
              variant={sortOrder === 'correlation' ? 'primary' : 'secondary'}
              onClick={() => setSortOrder('correlation')}
            >
              <Activity className="w-3 h-3 mr-1" />
              BY CORRELATION
            </MatrixButton>
            <MatrixButton
              size="sm"
              variant={sortOrder === 'asset' ? 'primary' : 'secondary'}
              onClick={() => setSortOrder('asset')}
            >
              BY ASSET
            </MatrixButton>
          </div>

          {selectedAssets.length === 0 && (
            <div className="text-matrix-green/70 text-sm font-mono">
              Click on assets to filter. Showing all {uniqueAssets.length} assets.
            </div>
          )}

          {selectedAssets.length > 0 && (
            <div className="flex gap-2 items-center">
              <span className="text-matrix-green/70 text-sm font-mono">
                Filtered: {selectedAssets.length} assets
              </span>
              <MatrixButton
                size="sm"
                variant="secondary"
                onClick={() => setSelectedAssets([])}
              >
                CLEAR FILTERS
              </MatrixButton>
            </div>
          )}

          {onExport && (
            <MatrixButton
              size="sm"
              variant="secondary"
              onClick={onExport}
              className="ml-auto"
            >
              EXPORT MATRIX
            </MatrixButton>
          )}
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-matrix-green/10 border border-matrix-green/30 rounded">
            <BarChart3 className="w-6 h-6 text-matrix-green mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Avg Correlation</p>
            <p className="text-matrix-green font-bold">
              {matrixStats.avgCorrelation.toFixed(3)}
            </p>
          </div>
          
          <div className="text-center p-3 bg-green-900/10 border border-green-500/30 rounded">
            <TrendingUp className="w-6 h-6 text-green-400 mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Max Correlation</p>
            <p className="text-green-400 font-bold">
              {matrixStats.maxCorrelation.toFixed(3)}
            </p>
          </div>
          
          <div className="text-center p-3 bg-red-900/10 border border-red-500/30 rounded">
            <TrendingDown className="w-6 h-6 text-red-400 mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Min Correlation</p>
            <p className="text-red-400 font-bold">
              {matrixStats.minCorrelation.toFixed(3)}
            </p>
          </div>
          
          <div className="text-center p-3 bg-yellow-900/10 border border-yellow-500/30 rounded">
            <div className="w-6 h-6 mx-auto mb-1 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-yellow-400" />
            </div>
            <p className="text-matrix-green/70 text-sm font-mono">High Corr. Pairs</p>
            <p className="text-yellow-400 font-bold">
              {matrixStats.highCorrelationPairs}/{matrixStats.totalPairs}
            </p>
          </div>
        </div>

        {/* Correlation Matrix */}
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="sticky left-0 bg-matrix-black p-2 text-matrix-green font-mono text-xs border-r border-matrix-green/30 z-10">
                    Assets
                  </th>
                  {filteredAssets.map(asset => (
                    <th
                      key={asset}
                      className={`p-2 text-matrix-green font-mono text-xs border-r border-matrix-green/30 min-w-[60px] cursor-pointer hover:bg-matrix-green/10 ${
                        selectedAssets.includes(asset) ? 'bg-matrix-green/20' : ''
                      }`}
                      onClick={() => {
                        if (interactive) {
                          setSelectedAssets(prev => 
                            prev.includes(asset) 
                              ? prev.filter(a => a !== asset)
                              : [...prev, asset]
                          );
                        }
                      }}
                    >
                      <div className="transform -rotate-45 origin-center whitespace-nowrap">
                        {getAssetName(asset)}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredAssets.map(asset1 => (
                  <tr key={asset1}>
                    <td
                      className={`sticky left-0 bg-matrix-black p-2 text-matrix-green font-mono text-sm border-r border-matrix-green/30 border-b border-matrix-green/10 z-10 cursor-pointer hover:bg-matrix-green/10 ${
                        selectedAssets.includes(asset1) ? 'bg-matrix-green/20' : ''
                      }`}
                      onClick={() => {
                        if (interactive) {
                          setSelectedAssets(prev => 
                            prev.includes(asset1) 
                              ? prev.filter(a => a !== asset1)
                              : [...prev, asset1]
                          );
                        }
                      }}
                    >
                      {getAssetName(asset1)}
                    </td>
                    {filteredAssets.map(asset2 => {
                      const correlation = correlationMatrix[asset1][asset2];
                      const color = getCorrelationColor(correlation);
                      const intensity = getCorrelationIntensity(correlation);
                      const isHighlighted = Math.abs(correlation) >= correlationThreshold;
                      
                      return (
                        <td
                          key={`${asset1}-${asset2}`}
                          className={`
                            relative p-1 border border-matrix-green/10 text-center cursor-pointer
                            hover:scale-105 hover:z-10 transition-all
                            ${isHighlighted ? 'ring-2 ring-yellow-400' : ''}
                          `}
                          style={{
                            backgroundColor: color,
                            opacity: 0.3 + intensity * 0.5,
                          }}
                          onMouseEnter={() => setHoveredCell({ asset1, asset2 })}
                          onMouseLeave={() => setHoveredCell(null)}
                          onClick={() => handleCellClick(asset1, asset2)}
                          title={`${asset1} vs ${asset2}: ${correlation.toFixed(3)} (${getCorrelationStrength(correlation)})`}
                        >
                          {showValues && (
                            <span className="text-xs font-mono font-bold text-black drop-shadow-sm">
                              {correlation.toFixed(2)}
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span className="text-red-400">Strong Negative (&lt;-0.8)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-500 rounded"></div>
            <span className="text-orange-400">Medium Negative (-0.8 to -0.5)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span className="text-yellow-400">Weak Negative (-0.5 to -0.2)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-500 rounded"></div>
            <span className="text-gray-400">Neutral (-0.2 to 0.2)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-lime-500 rounded"></div>
            <span className="text-lime-400">Weak Positive (0.2 to 0.5)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span className="text-green-400">Strong Positive (&gt;0.8)</span>
          </div>
        </div>

        {/* Hover Info */}
        {hoveredCell && (
          <div className="bg-matrix-black border border-matrix-green rounded-lg p-4">
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-matrix-green font-mono font-bold">
                  {getAssetName(hoveredCell.asset1)}
                </p>
                <p className="text-matrix-green/70 text-sm font-mono">vs</p>
                <p className="text-matrix-green font-mono font-bold">
                  {getAssetName(hoveredCell.asset2)}
                </p>
              </div>
              
              <div>
                <p className="text-matrix-green/70 font-mono text-sm">Correlation</p>
                <p className={`font-mono font-bold ${
                  correlationMatrix[hoveredCell.asset1][hoveredCell.asset2] >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {correlationMatrix[hoveredCell.asset1][hoveredCell.asset2].toFixed(4)}
                </p>
              </div>
              
              <div>
                <p className="text-matrix-green/70 font-mono text-sm">Strength</p>
                <p className="text-matrix-green font-mono font-bold">
                  {getCorrelationStrength(correlationMatrix[hoveredCell.asset1][hoveredCell.asset2])}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </MatrixCard>
  );
};

export default CorrelationMatrix;