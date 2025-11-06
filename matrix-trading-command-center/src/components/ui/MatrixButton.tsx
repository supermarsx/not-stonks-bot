import React from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface MatrixButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  glow?: boolean;
}

export const MatrixButton: React.FC<MatrixButtonProps> = ({
  children,
  className,
  variant = 'primary',
  size = 'md',
  loading = false,
  glow = false,
  disabled,
  ...props
}) => {
  const baseClasses = 'matrix-button font-mono font-bold transition-all duration-300 border-2 relative overflow-hidden';
  
  const variants = {
    primary: 'border-green-700 bg-green-900/30 text-green-400 hover:bg-green-600 hover:text-black',
    secondary: 'border-gray-700 bg-gray-800/30 text-gray-300 hover:bg-gray-600 hover:text-black',
    danger: 'border-red-700 bg-red-900/30 text-red-400 hover:bg-red-600 hover:text-black',
    success: 'border-green-700 bg-green-900/30 text-green-400 hover:bg-green-600 hover:text-black',
  };
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
  };
  
  return (
    <motion.button
      className={cn(
        baseClasses,
        variants[variant],
        sizes[size],
        glow && 'matrix-border-glow',
        disabled && 'opacity-50 cursor-not-allowed',
        className
      )}
      disabled={disabled || loading}
      whileHover={!disabled && !loading ? { scale: 1.02 } : {}}
      whileTap={!disabled && !loading ? { scale: 0.98 } : {}}
      {...props}
    >
      {loading && (
        <motion.div
          className="absolute inset-0 bg-green-400/20"
          animate={{ opacity: [0, 1, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
      
      <span className="relative z-10 flex items-center gap-2">
        {loading && (
          <motion.div
            className="w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          />
        )}
        {children}
      </span>
    </motion.button>
  );
};