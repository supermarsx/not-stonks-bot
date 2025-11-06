import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData, HistogramData, BarData } from 'lightweight-charts';
import { MatrixCard } from '../MatrixCard';

interface TradingViewChartProps {
  data: any[];
  chartType: 'line' | 'candlestick' | 'bar' | 'histogram';
  indicators?: {
    sma?: { period: number; color: string }[];
    ema?: { period: number; color: string }[];
    rsi?: { period: number; color: string };
    macd?: { fast: number; slow: number; signal: number };
    bollinger?: { period: number; deviation: number };
  };
  height?: number;
  showGrid?: boolean;
  showVolume?: boolean;
  theme?: 'dark' | 'light';
  onPriceChange?: (price: number) => void;
  onCrosshairMove?: (data: any) => void;
}

export const TradingViewChart: React.FC<TradingViewChartProps> = ({
  data,
  chartType,
  indicators,
  height = 400,
  showGrid = true,
  showVolume = false,
  theme = 'dark',
  onPriceChange,
  onCrosshairMove,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<any> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<any> | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  // Theme colors
  const colors = theme === 'dark' ? {
    background: '#0a0a0a',
    textColor: '#00ff00',
    gridColor: '#00ff0033',
    borderColor: '#00ff00',
    upColor: '#00ff00',
    downColor: '#ff0000',
    volumeColor: '#00ff0033',
  } : {
    background: '#ffffff',
    textColor: '#000000',
    gridColor: '#00000022',
    borderColor: '#000000',
    upColor: '#26a69a',
    downColor: '#ef5350',
    volumeColor: '#00000033',
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height,
      layout: {
        background: { color: colors.background },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: colors.borderColor,
      },
      timeScale: {
        borderColor: colors.borderColor,
        timeVisible: true,
        secondsVisible: false,
      },
      watermark: {
        visible: true,
        fontSize: 48,
        horzAlign: 'center',
        vertAlign: 'center',
        color: colors.textColor,
        text: 'Matrix Trading',
      },
    });

    // Create main series
    let series;
    switch (chartType) {
      case 'line':
        series = chart.addLineSeries({
          color: colors.upColor,
          lineWidth: 2,
        });
        break;
      case 'candlestick':
        series = chart.addCandlestickSeries({
          upColor: colors.upColor,
          downColor: colors.downColor,
          borderDownColor: colors.downColor,
          borderUpColor: colors.upColor,
          wickDownColor: colors.downColor,
          wickUpColor: colors.upColor,
        });
        break;
      case 'bar':
        series = chart.addBarSeries({
          upColor: colors.upColor,
          downColor: colors.downColor,
        });
        break;
      case 'histogram':
        series = chart.addHistogramSeries({
          color: colors.volumeColor,
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        });
        break;
    }

    // Create volume series if enabled
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        color: colors.volumeColor,
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
      volumeSeriesRef.current = volumeSeries;
    }

    seriesRef.current = series;
    chartRef.current = chart;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        const width = chartContainerRef.current.clientWidth;
        setContainerWidth(width);
        chart.applyOptions({ width });
      }
    };

    // Set initial width
    handleResize();
    window.addEventListener('resize', handleResize);

    // Crosshair move handler
    chart.subscribeCrosshairMove((param) => {
      if (param.seriesPrices.size > 0 && onCrosshairMove) {
        const seriesData = param.seriesPrices.get(series);
        if (seriesData) {
          onCrosshairMove({
            price: seriesData,
            time: param.time,
            position: param.point,
          });
        }
      }
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [chartType, height, theme, showVolume]);

  // Update chart data
  useEffect(() => {
    if (seriesRef.current && data.length > 0) {
      try {
        seriesRef.current.setData(data);
        
        // Add volume data if enabled
        if (volumeSeriesRef.current && showVolume) {
          const volumeData = data.map(item => ({
            time: item.time,
            value: item.volume || 0,
            color: item.close >= item.open ? colors.upColor : colors.downColor,
          }));
          volumeSeriesRef.current.setData(volumeData);
        }
      } catch (error) {
        console.error('Error setting chart data:', error);
      }
    }
  }, [data, showVolume, colors]);

  // Add technical indicators
  useEffect(() => {
    if (!chartRef.current || !indicators || !data.length) return;

    // Clear existing indicators (simplified approach)
    // In a real implementation, you'd want to track and manage indicators properly

    // Add SMA indicators
    indicators.sma?.forEach((sma, index) => {
      const smaData = calculateSMA(data, sma.period);
      const smaSeries = chartRef.current!.addLineSeries({
        color: sma.color,
        lineWidth: 1,
        title: `SMA ${sma.period}`,
      });
      smaSeries.setData(smaData);
    });

    // Add EMA indicators
    indicators.ema?.forEach((ema, index) => {
      const emaData = calculateEMA(data, ema.period);
      const emaSeries = chartRef.current!.addLineSeries({
        color: ema.color,
        lineWidth: 1,
        title: `EMA ${ema.period}`,
      });
      emaSeries.setData(emaData);
    });

  }, [indicators, data]);

  return (
    <MatrixCard className="p-4">
      <div
        ref={chartContainerRef}
        style={{ width: '100%', height: `${height}px` }}
        className="chart-container"
      />
    </MatrixCard>
  );
};

// Helper functions for technical indicators
function calculateSMA(data: any[], period: number): LineData[] {
  const result: LineData[] = [];
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((acc, item) => acc + item.close, 0);
    const average = sum / period;
    result.push({
      time: data[i].time,
      value: average,
    });
  }
  return result;
}

function calculateEMA(data: any[], period: number): LineData[] {
  const result: LineData[] = [];
  const multiplier = 2 / (period + 1);
  let ema = data[0].close;

  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      result.push({
        time: data[i].time,
        value: ema,
      });
    } else {
      ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
      result.push({
        time: data[i].time,
        value: ema,
      });
    }
  }
  return result;
}

export default TradingViewChart;
