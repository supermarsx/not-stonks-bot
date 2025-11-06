import { cn } from '../utils/cn';

interface StatCardProps {
  label: string;
  value: string | number;
  change?: number;
  changePercent?: number;
  icon?: React.ReactNode;
  loading?: boolean;
  className?: string;
}

export function StatCard({
  label,
  value,
  change,
  changePercent,
  icon,
  loading = false,
  className,
}: StatCardProps) {
  const isPositive = (change ?? 0) >= 0;
  const hasChange = change !== undefined;

  if (loading) {
    return (
      <div className={cn('border-2 border-matrix-green bg-matrix-black p-4', className)}>
        <div className="animate-pulse space-y-2">
          <div className="h-4 w-20 bg-matrix-green/30"></div>
          <div className="h-8 w-32 bg-matrix-green/30"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('border-2 border-matrix-green bg-matrix-black p-4 shadow-[0_0_10px_rgba(0,255,0,0.2)]', className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs text-matrix-green/70 uppercase tracking-wider">{label}</p>
          <p className="mt-2 text-2xl font-bold text-matrix-green font-mono">{value}</p>
          {hasChange && (
            <p className={cn('mt-1 text-sm font-bold', isPositive ? 'text-matrix-green' : 'text-red-500')}>
              {isPositive ? '+' : ''}
              {change?.toFixed(2)}
              {changePercent !== undefined && ` (${isPositive ? '+' : ''}${changePercent.toFixed(2)}%)`}
            </p>
          )}
        </div>
        {icon && (
          <div className="text-matrix-green/50">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
