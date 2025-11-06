import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

interface MatrixRainProps {
  density?: number; // Number of columns (0-1, where 1 is maximum density)
  speed?: number; // Animation speed multiplier
  color?: string; // Color of the rain (default: matrix green)
  className?: string;
}

export const MatrixRain: React.FC<MatrixRainProps> = ({
  density = 0.3,
  speed = 1,
  color = '#00ff41',
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const dropsRef = useRef<number[]>([]);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const updateCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      
      // Initialize drops
      const columns = Math.floor(canvas.width / 20);
      dropsRef.current = new Array(columns).fill(0).map(() => 
        Math.floor(Math.random() * canvas.height / 20)
      );
    };
    
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    
    const matrixChars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
    
    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      ctx.fillStyle = color;
      ctx.font = '15px "Courier New", monospace';
      
      dropsRef.current.forEach((drop, i) => {
        const text = matrixChars[Math.floor(Math.random() * matrixChars.length)];
        const x = i * 20;
        const y = drop * 20;
        
        // Add glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fillText(text, x, y);
        ctx.shadowBlur = 0;
        
        // Reset drop when it goes off screen
        if (y > canvas.height && Math.random() > 0.975) {
          dropsRef.current[i] = 0;
        } else {
          dropsRef.current[i]++;
        }
      });
    };
    
    const animate = () => {
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('resize', updateCanvasSize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [density, speed, color]);
  
  return (
    <motion.canvas
      ref={canvasRef}
      className={`fixed inset-0 pointer-events-none z-0 ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 2 }}
    />
  );
};

// Simple CSS-based matrix rain effect
export const SimpleMatrixRain: React.FC<{
  density?: number;
  className?: string;
}> = ({ density = 0.3, className = '' }) => {
  const characters = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
  
  const rainChars = Array.from({ length: Math.floor(50 * density) }, (_, i) => (
    <motion.div
      key={i}
      className="absolute text-green-500 font-mono text-sm opacity-70"
      style={{
        left: `${Math.random() * 100}%`,
        fontSize: `${12 + Math.random() * 8}px`,
      }}
      animate={{
        y: ['-20px', 'calc(100vh + 20px)'],
        opacity: [0, 0.8, 0.2, 0],
      }}
      transition={{
        duration: 2 + Math.random() * 3,
        repeat: Infinity,
        delay: Math.random() * 5,
        ease: 'linear',
      }}
    >
      {characters[Math.floor(Math.random() * characters.length)]}
    </motion.div>
  ));
  
  return (
    <div className={`matrix-rain-container ${className}`}>
      {rainChars}
    </div>
  );
};