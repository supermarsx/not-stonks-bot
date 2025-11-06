import { cn } from '../utils/cn';

interface PriceDisplayProps {
  value: number;
  change?: number;
  changePercent?: number;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  currency?: string;
  showSign?: boolean;
  className?: string;
}

export function PriceDisplay({
  value,
  change,
  changePercent,
  size = 'md',
  currency = '$',
  showSign = false,
  className,
}: PriceDisplayProps) {
  const isPositive = (change ?? 0) >= 0;
  const displayChange = change !== undefined;

  const sizes = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-xl',
    xl: 'text-3xl',
  };

  const changeColor = isPositive ? 'text-matrix-green' : 'text-red-500';

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <span className={cn('font-mono font-bold', sizes[size])}>
        {currency}
        {Math.abs(value).toLocaleString('en-US', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}
      </span>
      {displayChange && (
        <span className={cn('text-sm font-bold', changeColor)}>
          {isPositive && showSign && '+'}
          {change?.toFixed(2)}
          {changePercent !== undefined && ` (${isPositive ? '+' : ''}${changePercent.toFixed(2)}%)`}
        </span>
      )}
    </div>
  );
}
