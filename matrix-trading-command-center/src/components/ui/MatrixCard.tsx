import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface MatrixCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  glow?: boolean;
  pulse?: boolean;
  interactive?: boolean;
}

export const MatrixCard: React.FC<MatrixCardProps> = ({
  children,
  className,
  title,
  subtitle,
  glow = false,
  pulse = false,
  interactive = false,
  ...props
}) => {
  return (
    <motion.div
      className={cn(
        'matrix-card p-6 relative overflow-hidden',
        glow && 'matrix-border-glow',
        pulse && 'animate-pulse',
        interactive && 'cursor-pointer hover:scale-[1.02] transition-transform',
        className
      )}
      whileHover={interactive ? { y: -2 } : {}}
      {...props}
    >
      {/* Scan line effect */}
      <div className="absolute inset-0 matrix-scan-line opacity-20 pointer-events-none" />
      
      {/* Content */}
      <div className="relative z-10">
        {(title || subtitle) && (
          <div className="mb-4 pb-3 border-b border-green-800/30">
            {title && (
              <h3 className="text-lg font-bold matrix-text-glow text-green-400">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-sm text-green-600 mt-1">{subtitle}</p>
            )}
          </div>
        )}
        
        {children}
      </div>
      
      {/* Corner decorations */}
      <div className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-green-500/50" />
      <div className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-green-500/50" />
      <div className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-green-500/50" />
      <div className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-green-500/50" />
    </motion.div>
  );
};