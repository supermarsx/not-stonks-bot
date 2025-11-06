declare module 'recharts' {
  import { Component } from 'react';

  export interface CartesianGridProps {
    stroke?: string;
    strokeDasharray?: string;
    horizontal?: boolean;
    vertical?: boolean;
  }

  export interface XAxisProps {
    dataKey?: string;
    type?: 'number' | 'category';
    stroke?: string;
    tick?: any;
    tickFormatter?: (value: any) => string;
    name?: string;
    height?: number;
    interval?: number | 'preserveStart' | 'preserveEnd' | 'preserveStartEnd';
    textAnchor?: string;
    angle?: number;
    yAxisId?: string | number;
  }

  export interface YAxisProps {
    stroke?: string;
    tick?: any;
    tickFormatter?: (value: any) => string;
    name?: string;
    yAxisId?: string | number;
    dataKey?: string;
    type?: 'number' | 'category';
    angle?: number;
    orientation?: 'left' | 'right';
  }

  export interface TooltipProps<TValue = any, TName = any> {
    contentStyle?: React.CSSProperties;
    labelStyle?: React.CSSProperties;
    formatter?: (value: TValue, name: TName, props: any) => [React.ReactNode, React.ReactNode];
    labelFormatter?: (label: any, payload?: any) => React.ReactNode;
    titleFormatter?: (title: any, payload?: any) => React.ReactNode;
  }

  export interface LegendProps {
    wrapperStyle?: React.CSSProperties;
  }

  export interface LineProps {
    type?: 'basis' | 'basisClosed' | 'basisOpen' | 'linear' | 'linearClosed' | 'natural' | 'monotoneX' | 'monotoneY' | 'monotone' | 'step' | 'stepBefore' | 'stepAfter';
    dataKey: string;
    stroke?: string;
    strokeWidth?: number;
    fill?: string;
    name?: string;
    dot?: boolean | any;
    yAxisId?: string | number;
  }

  export interface AreaProps {
    type?: 'basis' | 'basisClosed' | 'basisOpen' | 'linear' | 'linearClosed' | 'natural' | 'monotoneX' | 'monotoneY' | 'monotone' | 'step' | 'stepBefore' | 'stepAfter';
    dataKey: string;
    stroke?: string;
    strokeWidth?: number;
    fill?: string;
    name?: string;
    opacity?: number;
  }

  export interface BarProps {
    dataKey: string;
    fill?: string;
    opacity?: number;
    name?: string;
    yAxisId?: string | number;
  }

  export interface ScatterProps {
    dataKey?: string;
    fill?: string;
    opacity?: number;
    name?: string;
    shape?: any;
    data?: any[];
  }

  export interface PieProps {
    data: any[];
    dataKey: string;
    cx?: string | number;
    cy?: string | number;
    outerRadius?: number;
    innerRadius?: number;
    fill?: string;
    startAngle?: number;
    endAngle?: number;
    labelLine?: boolean;
    label?: any;
    children?: React.ReactNode;
  }

  export interface CellProps {
    fill?: string;
  }

  export interface PolarGridProps {
    stroke?: string;
  }

  export interface PolarAngleAxisProps {
    dataKey: string;
    tick?: any;
  }

  export interface PolarRadiusAxisProps {
    angle?: number;
    domain?: [number, number];
    tick?: any;
  }

  export interface RadarProps {
    name: string;
    dataKey: string;
    stroke?: string;
    fill?: string;
    fillOpacity?: number;
    strokeWidth?: number;
  }

  export class ResponsiveContainer extends Component<{
    width?: string | number;
    height?: string | number;
    children: React.ReactNode;
  }> {}

  export class LineChart extends Component<{
    width?: number;
    height?: number;
    data: any[];
    children?: React.ReactNode;
  }> {}

  export class AreaChart extends Component<{
    width?: number;
    height?: number;
    data: any[];
    children?: React.ReactNode;
  }> {}

  export class BarChart extends Component<{
    width?: number;
    height?: number;
    data: any[];
    children?: React.ReactNode;
  }> {}

  export class ComposedChart extends Component<{
    width?: number;
    height?: number;
    data: any[];
    children?: React.ReactNode;
  }> {}

  export class ScatterChart extends Component<{
    width?: number;
    height?: number;
    data?: any[];
    children?: React.ReactNode;
  }> {}

  export class PieChart extends Component<{
    width?: number;
    height?: number;
    children?: React.ReactNode;
  }> {}

  export class RadarChart extends Component<{
    width?: number;
    height?: number;
    data: any[];
    children?: React.ReactNode;
  }> {}

  export class CartesianGrid extends Component<CartesianGridProps> {}
  export class XAxis extends Component<XAxisProps> {}
  export class YAxis extends Component<YAxisProps> {}
  export class Tooltip extends Component<TooltipProps> {}
  export class Legend extends Component<LegendProps> {}
  export class Line extends Component<LineProps> {}
  export class Area extends Component<AreaProps> {}
  export class Bar extends Component<BarProps> {}
  export class Scatter extends Component<ScatterProps> {}
  export class Pie extends Component<PieProps> {}
  export class Cell extends Component<CellProps> {}
  export class PolarGrid extends Component<PolarGridProps> {}
  export class PolarAngleAxis extends Component<PolarAngleAxisProps> {}
  export class PolarRadiusAxis extends Component<PolarRadiusAxisProps> {}
  export class Radar extends Component<RadarProps> {}
}