import { cn } from '../utils/cn';

type StatusType = 'success' | 'warning' | 'error' | 'info' | 'active' | 'inactive';

interface StatusBadgeProps {
  status: StatusType | string;
  className?: string;
}

const statusConfig: Record<StatusType, { color: string; text: string }> = {
  success: { color: 'bg-matrix-green text-matrix-black', text: 'SUCCESS' },
  warning: { color: 'bg-yellow-500 text-black', text: 'WARNING' },
  error: { color: 'bg-red-500 text-white', text: 'ERROR' },
  info: { color: 'bg-blue-500 text-white', text: 'INFO' },
  active: { color: 'bg-matrix-green text-matrix-black', text: 'ACTIVE' },
  inactive: { color: 'bg-gray-500 text-white', text: 'INACTIVE' },
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const normalizedStatus = status.toLowerCase() as StatusType;
  const config = statusConfig[normalizedStatus] || {
    color: 'bg-matrix-green text-matrix-black',
    text: status.toUpperCase(),
  };

  return (
    <span
      className={cn(
        'inline-block px-2 py-1 text-xs font-bold border-2 border-current',
        config.color,
        className
      )}
    >
      {config.text}
    </span>
  );
}
