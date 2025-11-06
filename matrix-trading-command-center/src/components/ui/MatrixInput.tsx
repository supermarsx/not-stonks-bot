import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface MatrixInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: React.ReactNode;
  glow?: boolean;
}

export const MatrixInput: React.FC<MatrixInputProps> = ({
  children,
  className,
  label,
  error,
  icon,
  glow = false,
  id,
  ...props
}) => {
  const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
  
  return (
    <div className="space-y-2">
      {label && (
        <label
          htmlFor={inputId}
          className="block text-sm font-mono font-bold text-green-400"
        >
          {label}
        </label>
      )}
      
      <div className="relative">
        {icon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-green-500">
            {icon}
          </div>
        )}
        
        <motion.input
          id={inputId}
          className={cn(
            'matrix-input w-full px-3 py-2 font-mono',
            icon && 'pl-10',
            glow && 'matrix-border-glow',
            error && 'border-red-500 focus:border-red-400',
            className
          )}
          whileFocus={{ scale: 1.01 }}
          {...props}
        />
        
        {/* Terminal cursor effect */}
        <motion.div
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-green-400 font-mono"
          animate={{ opacity: [0, 1, 0] }}
          transition={{ duration: 1, repeat: Infinity }}
        >
          |
        </motion.div>
      </div>
      
      {error && (
        <motion.p
          className="text-red-400 text-sm font-mono"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {error}
        </motion.p>
      )}
    </div>
  );
};