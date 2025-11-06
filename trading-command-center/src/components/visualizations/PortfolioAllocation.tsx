import React, { useState } from 'react';
import { ChartJSCanvas, createPortfolioChartData } from '../charts/ChartJSCanvas';
import { MatrixCard } from '../MatrixCard';
import { MatrixButton } from '../MatrixButton';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react';

interface Position {
  symbol: string;
  size: number;
  marketValue: number;
  allocation: number;
  pnl: number;
  pnlPercentage: number;
  side: 'long' | 'short';
}

interface PortfolioAllocationProps {
  positions: Position[];
  totalValue: number;
  title?: string;
  className?: string;
  showPnL?: boolean;
  viewType?: 'pie' | 'donut' | 'bar';
}

const COLORS = [
  '#00ff00',
  '#00cc00', 
  '#009900',
  '#006600',
  '#003300',
  '#336633',
  '#669966',
  '#99cc99',
  '#ccffcc',
  '#ffffff',
];

export const PortfolioAllocation: React.FC<PortfolioAllocationProps> = ({
  positions,
  totalValue,
  title = "Portfolio Allocation",
  className = "",
  showPnL = true,
  viewType = 'donut',
}) => {
  const [selectedView, setSelectedView] = useState<'pie' | 'donut' | 'bar' | 'table'>('donut');

  // Sort positions by allocation
  const sortedPositions = [...positions].sort((a, b) => b.allocation - a.allocation);

  // Prepare chart data
  const chartData = sortedPositions.map((position, index) => ({
    ...position,
    color: COLORS[index % COLORS.length],
  }));

  // Custom tooltip for the chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-matrix-black border border-matrix-green p-3 rounded shadow-lg">
          <p className="text-matrix-green font-mono">{data.symbol}</p>
          <p className="text-matrix-green">Allocation: {data.allocation.toFixed(2)}%</p>
          <p className="text-matrix-green">
            Value: ${data.marketValue.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </p>
          {showPnL && (
            <>
              <p className={`font-mono ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                P&L: ${data.pnl.toFixed(2)}
              </p>
              <p className={`font-mono ${data.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                P&L %: {data.pnlPercentage.toFixed(2)}%
              </p>
            </>
          )}
        </div>
      );
    }
    return null;
  };

  // Render pie/donut chart
  const renderPieChart = () => (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={viewType === 'donut' ? 60 : 0}
          outerRadius={100}
          paddingAngle={2}
          dataKey="allocation"
          nameKey="symbol"
        >
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ color: '#00ff00', fontFamily: 'monospace' }}
          formatter={(value) => <span style={{ color: '#00ff00' }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );

  // Render bar chart
  const renderBarChart = () => (
    <div className="space-y-2">
      {chartData.map((position, index) => (
        <div key={position.symbol} className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: position.color }}
            />
            <span className="text-matrix-green font-mono">{position.symbol}</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-32 h-3 bg-matrix-green/20 rounded-full overflow-hidden">
              <div
                className="h-full bg-matrix-green rounded-full"
                style={{ width: `${Math.min(position.allocation * 2, 100)}%` }}
              />
            </div>
            <span className="text-matrix-green font-mono text-sm w-16 text-right">
              {position.allocation.toFixed(1)}%
            </span>
            <span className="text-matrix-green font-mono text-sm w-24 text-right">
              ${(position.marketValue / 1000).toFixed(1)}k
            </span>
          </div>
        </div>
      ))}
    </div>
  );

  // Render table view
  const renderTable = () => (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-matrix-green/30">
            <th className="text-left py-2 text-matrix-green font-mono">Symbol</th>
            <th className="text-right py-2 text-matrix-green font-mono">Size</th>
            <th className="text-right py-2 text-matrix-green font-mono">Market Value</th>
            <th className="text-right py-2 text-matrix-green font-mono">Allocation</th>
            {showPnL && (
              <>
                <th className="text-right py-2 text-matrix-green font-mono">P&L</th>
                <th className="text-right py-2 text-matrix-green font-mono">P&L %</th>
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {sortedPositions.map((position, index) => (
            <tr key={position.symbol} className="border-b border-matrix-green/10">
              <td className="py-2">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-matrix-green font-mono">{position.symbol}</span>
                  <span className={`text-xs px-1 rounded ${
                    position.side === 'long' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'
                  }`}>
                    {position.side.toUpperCase()}
                  </span>
                </div>
              </td>
              <td className="text-right py-2 text-matrix-green font-mono">
                {position.size.toFixed(4)}
              </td>
              <td className="text-right py-2 text-matrix-green font-mono">
                ${position.marketValue.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </td>
              <td className="text-right py-2 text-matrix-green font-mono">
                {position.allocation.toFixed(2)}%
              </td>
              {showPnL && (
                <>
                  <td className={`text-right py-2 font-mono ${
                    position.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    ${position.pnl.toFixed(2)}
                  </td>
                  <td className={`text-right py-2 font-mono ${
                    position.pnlPercentage >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {position.pnlPercentage.toFixed(2)}%
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <MatrixCard title={title} className={className}>
      <div className="space-y-4">
        {/* View Controls */}
        <div className="flex gap-2 mb-4">
          {(['pie', 'donut', 'bar', 'table'] as const).map((view) => (
            <MatrixButton
              key={view}
              size="sm"
              variant={selectedView === view ? 'primary' : 'secondary'}
              onClick={() => setSelectedView(view)}
            >
              {view.toUpperCase()}
            </MatrixButton>
          ))}
        </div>

        {/* Portfolio Summary */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center">
            <DollarSign className="w-6 h-6 text-matrix-green mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Total Value</p>
            <p className="text-matrix-green font-bold">
              ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </p>
          </div>
          <div className="text-center">
            <TrendingUp className="w-6 h-6 text-matrix-green mx-auto mb-1" />
            <p className="text-matrix-green/70 text-sm font-mono">Total Positions</p>
            <p className="text-matrix-green font-bold">{positions.length}</p>
          </div>
          <div className="text-center">
            <div className="w-6 h-6 mx-auto mb-1 flex items-center justify-center">
              <div className="w-3 h-3 rounded-full bg-matrix-green mr-1" />
              <div className="w-3 h-3 rounded-full bg-red-500 mr-1" />
            </div>
            <p className="text-matrix-green/70 text-sm font-mono">Long/Short</p>
            <p className="text-matrix-green font-bold">
              {positions.filter(p => p.side === 'long').length}/
              {positions.filter(p => p.side === 'short').length}
            </p>
          </div>
        </div>

        {/* Chart/Table View */}
        {selectedView === 'pie' || selectedView === 'donut' ? (
          <div className="h-80">
            {renderPieChart()}
          </div>
        ) : selectedView === 'bar' ? (
          <div className="h-80 overflow-y-auto">
            {renderBarChart()}
          </div>
        ) : (
          <div className="max-h-80 overflow-y-auto">
            {renderTable()}
          </div>
        )}
      </div>
    </MatrixCard>
  );
};

export default PortfolioAllocation;