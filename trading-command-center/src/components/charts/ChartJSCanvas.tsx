import React, { useRef, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler,
  ChartOptions,
} from 'chart.js';
import { Line, Bar, Doughnut, Pie, Bubble, Scatter, PolarArea } from 'react-chartjs-2';
import { MatrixCard } from '../MatrixCard';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
);

interface ChartJSCanvasProps {
  type: 'line' | 'bar' | 'doughnut' | 'pie' | 'bubble' | 'scatter' | 'polarArea';
  data: any;
  options?: ChartOptions;
  height?: number;
  width?: number;
  plugins?: any[];
  onClick?: (event: any, elements: any[]) => void;
  onHover?: (event: any, elements: any[]) => void;
  className?: string;
  title?: string;
  subtitle?: string;
}

export const ChartJSCanvas: React.FC<ChartJSCanvasProps> = ({
  type,
  data,
  options,
  height = 400,
  width = 800,
  plugins = [],
  onClick,
  onHover,
  className = '',
  title,
  subtitle,
}) => {
  const chartRef = useRef<ChartJS | null>(null);

  // Default matrix theme
  const defaultOptions: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#00ff00',
          font: {
            family: 'monospace',
          },
        },
      },
      title: {
        display: !!title,
        text: title,
        color: '#00ff00',
        font: {
          family: 'monospace',
          size: 16,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#00ff00',
        bodyColor: '#00ff00',
        borderColor: '#00ff00',
        borderWidth: 1,
      },
    },
    scales: type !== 'doughnut' && type !== 'pie' && type !== 'polarArea' && type !== 'bubble' && type !== 'scatter' ? {
      x: {
        grid: {
          color: '#00ff0033',
        },
        ticks: {
          color: '#00ff00',
          font: {
            family: 'monospace',
          },
        },
      },
      y: {
        grid: {
          color: '#00ff0033',
        },
        ticks: {
          color: '#00ff00',
          font: {
            family: 'monospace',
          },
        },
      },
    } : {},
    animation: {
      duration: 750,
      easing: 'easeInOutQuart',
    },
    interaction: {
      intersect: false,
      mode: 'index',
    },
  };

  const mergedOptions = {
    ...defaultOptions,
    ...options,
    plugins: {
      ...defaultOptions.plugins,
      ...options?.plugins,
    },
  };

  // Chart component mapping
  const ChartComponent = {
    line: Line,
    bar: Bar,
    doughnut: Doughnut,
    pie: Pie,
    bubble: Bubble,
    scatter: Scatter,
    polarArea: PolarArea,
  }[type];

  const chartOptions = {
    ...mergedOptions,
    onClick,
    onHover,
  };

  return (
    <MatrixCard 
      title={title} 
      subtitle={subtitle} 
      className={`p-4 ${className}`}
    >
      <div style={{ height: `${height}px`, width: '100%' }}>
        <ChartComponent
          ref={chartRef}
          data={data}
          options={chartOptions}
          plugins={plugins}
        />
      </div>
    </MatrixCard>
  );
};

// Utility function to create sample portfolio data
export const createPortfolioChartData = (portfolio: any[]) => {
  return {
    labels: portfolio.map(item => item.symbol || item.name),
    datasets: [
      {
        label: 'Portfolio Allocation',
        data: portfolio.map(item => item.allocation || item.value),
        backgroundColor: [
          '#00ff00',
          '#00cc00', 
          '#009900',
          '#006600',
          '#003300',
          '#336633',
          '#669966',
          '#99cc99',
        ],
        borderColor: '#00ff00',
        borderWidth: 1,
      },
    ],
  };
};

// Utility function to create performance chart data
export const createPerformanceChartData = (data: any[]) => {
  return {
    labels: data.map(item => item.date),
    datasets: [
      {
        label: 'Portfolio Value',
        data: data.map(item => item.value),
        borderColor: '#00ff00',
        backgroundColor: 'rgba(0, 255, 0, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Benchmark',
        data: data.map(item => item.benchmark || item.value * 0.95),
        borderColor: '#ffff00',
        backgroundColor: 'rgba(255, 255, 0, 0.1)',
        fill: false,
        tension: 0.4,
      },
    ],
  };
};

// Utility function to create P&L waterfall data
export const createWaterfallChartData = (data: any[]) => {
  return {
    labels: data.map(item => item.label),
    datasets: [
      {
        label: 'P&L',
        data: data.map(item => item.value),
        backgroundColor: data.map(item => 
          item.value >= 0 ? '#00ff00' : '#ff0000'
        ),
        borderColor: data.map(item => 
          item.value >= 0 ? '#00cc00' : '#cc0000'
        ),
        borderWidth: 1,
      },
    ],
  };
};

// Utility function to create correlation heatmap data
export const createHeatmapData = (correlationMatrix: number[][]) => {
  const labels = correlationMatrix.map((_, index) => `Asset ${index + 1}`);
  
  const datasets = correlationMatrix.map((row, rowIndex) => ({
    label: labels[rowIndex],
    data: row,
    backgroundColor: row.map(value => {
      const intensity = Math.abs(value);
      if (value > 0) {
        return `rgba(0, 255, 0, ${intensity})`;
      } else {
        return `rgba(255, 0, 0, ${intensity})`;
      }
    }),
    borderColor: '#00ff00',
    borderWidth: 1,
  }));

  return {
    labels,
    datasets: [{
      label: 'Correlation Matrix',
      data: correlationMatrix.flat(),
      backgroundColor: correlationMatrix.flat().map(value => {
        const intensity = Math.abs(value);
        if (value > 0) {
          return `rgba(0, 255, 0, ${intensity})`;
        } else {
          return `rgba(255, 0, 0, ${intensity})`;
        }
      }),
    }],
  };
};

export default ChartJSCanvas;
