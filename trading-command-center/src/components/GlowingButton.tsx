import { ButtonHTMLAttributes, ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from '../utils/cn';

interface GlowingButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: ReactNode;
}

export function GlowingButton({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  className,
  disabled,
  ...props
}: GlowingButtonProps) {
  const baseStyles = 'border-2 font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2';
  
  const variants = {
    primary: 'border-matrix-green bg-matrix-green text-matrix-black hover:bg-matrix-black hover:text-matrix-green hover:shadow-[0_0_15px_rgba(0,255,0,0.5)]',
    secondary: 'border-matrix-green bg-matrix-black text-matrix-green hover:bg-matrix-green hover:text-matrix-black hover:shadow-[0_0_15px_rgba(0,255,0,0.5)]',
    danger: 'border-red-500 bg-red-500 text-white hover:bg-black hover:text-red-500 hover:shadow-[0_0_15px_rgba(255,0,0,0.5)]',
    ghost: 'border-transparent bg-transparent text-matrix-green hover:border-matrix-green hover:shadow-[0_0_10px_rgba(0,255,0,0.3)]',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  };

  return (
    <button
      className={cn(baseStyles, variants[variant], sizes[size], className)}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <Loader2 className="animate-spin" size={size === 'sm' ? 14 : size === 'lg' ? 20 : 16} />
      ) : icon ? (
        <span>{icon}</span>
      ) : null}
      {children}
    </button>
  );
}
