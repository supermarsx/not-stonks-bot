import { ReactNode } from 'react';
import { cn } from '../utils/cn';

interface MatrixCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  glow?: boolean;
  noPadding?: boolean;
}

export function MatrixCard({ 
  children, 
  className, 
  title, 
  subtitle, 
  glow = false,
  noPadding = false 
}: MatrixCardProps) {
  return (
    <div
      className={cn(
        'border-2 border-matrix-green bg-matrix-black',
        glow && 'shadow-[0_0_10px_rgba(0,255,0,0.3)]',
        className
      )}
    >
      {(title || subtitle) && (
        <div className="border-b-2 border-matrix-green px-4 py-3">
          {title && (
            <h3 className="text-lg font-bold matrix-glow-text">{title}</h3>
          )}
          {subtitle && (
            <p className="mt-1 text-sm text-matrix-green/70">{subtitle}</p>
          )}
        </div>
      )}
      <div className={noPadding ? '' : 'p-4'}>{children}</div>
    </div>
  );
}