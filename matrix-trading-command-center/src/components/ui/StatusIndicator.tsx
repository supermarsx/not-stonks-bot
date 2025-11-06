import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface StatusIndicatorProps {
  status: 'positive' | 'negative' | 'neutral' | 'warning' | 'online' | 'offline' | 'error';
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
  children?: React.ReactNode;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  size = 'md',
  pulse = false,
  children
}) => {
  const statusConfig = {
    positive: { 
      color: '#00ff41', 
      bgColor: 'bg-green-500/20', 
      borderColor: 'border-green-500',
      label: 'Positive'
    },
    negative: { 
      color: '#ff0040', 
      bgColor: 'bg-red-500/20', 
      borderColor: 'border-red-500',
      label: 'Negative'
    },
    neutral: { 
      color: '#ffff00', 
      bgColor: 'bg-yellow-500/20', 
      borderColor: 'border-yellow-500',
      label: 'Neutral'
    },
    warning: { 
      color: '#ffaa00', 
      bgColor: 'bg-orange-500/20', 
      borderColor: 'border-orange-500',
      label: 'Warning'
    },
    online: { 
      color: '#00ff41', 
      bgColor: 'bg-green-500/20', 
      borderColor: 'border-green-500',
      label: 'Online'
    },
    offline: { 
      color: '#808080', 
      bgColor: 'bg-gray-500/20', 
      borderColor: 'border-gray-500',
      label: 'Offline'
    },
    error: { 
      color: '#ff0040', 
      bgColor: 'bg-red-500/20', 
      borderColor: 'border-red-500',
      label: 'Error'
    }
  };
  
  const sizes = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };
  
  const config = statusConfig[status];
  
  return (
    <div className="flex items-center gap-2">
      <motion.div
        className={cn(
          'rounded-full border-2 flex-shrink-0',
          sizes[size],
          config.bgColor,
          config.borderColor
        )}
        animate={pulse ? {
          boxShadow: [
            `0 0 0 0 ${config.color}40`,
            `0 0 0 4px ${config.color}00`,
          ],
        } : {}}
        transition={{ duration: 2, repeat: pulse ? Infinity : 0 }}
      />
      
      {children && (
        <span 
          className="font-mono text-sm"
          style={{ color: config.color }}
        >
          {children}
        </span>
      )}
    </div>
  );
};