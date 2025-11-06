import React from 'react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface MatrixTableProps<T> {
  data: T[];
  columns: {
    key: keyof T;
    title: string;
    width?: string;
    align?: 'left' | 'center' | 'right';
    render?: (value: any, row: T, index: number) => React.ReactNode;
  }[];
  className?: string;
  loading?: boolean;
  emptyMessage?: string;
  hoverable?: boolean;
  sortable?: boolean;
  onSort?: (key: keyof T) => void;
}

export function MatrixTable<T extends Record<string, any>>({
  data,
  columns,
  className,
  loading = false,
  emptyMessage = 'No data available',
  hoverable = true,
  sortable = false,
  onSort
}: MatrixTableProps<T>) {
  if (loading) {
    return (
      <div className="matrix-card p-8 text-center">
        <motion.div
          className="inline-block w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <p className="mt-4 text-green-400 font-mono">Loading data...</p>
      </div>
    );
  }
  
  if (data.length === 0) {
    return (
      <div className="matrix-card p-8 text-center">
        <p className="text-green-600 font-mono">{emptyMessage}</p>
      </div>
    );
  }
  
  return (
    <div className={cn('matrix-card overflow-hidden', className)}>
      <div className="overflow-x-auto">
        <table className="matrix-table w-full">
          <thead>
            <tr>
              {columns.map((column, index) => (
                <th
                  key={String(column.key)}
                  className={cn(
                    'font-mono font-bold text-green-400 border-b-2 border-green-800',
                    column.align === 'center' && 'text-center',
                    column.align === 'right' && 'text-right',
                    sortable && 'cursor-pointer hover:bg-green-900/20 transition-colors'
                  )}
                  style={{ width: column.width }}
                  onClick={() => sortable && onSort?.(column.key)}
                >
                  <div className="flex items-center justify-between">
                    <span>{column.title}</span>
                    {sortable && (
                      <motion.span
                        className="ml-2 text-green-500"
                        animate={{ rotate: 0 }}
                      >
                        ↕
                      </motion.span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <AnimatePresence mode="popLayout">
              {data.map((row, rowIndex) => (
                <motion.tr
                  key={row.id || rowIndex}
                  className={cn(
                    'transition-all duration-200',
                    hoverable && 'hover:bg-green-900/10'
                  )}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ delay: rowIndex * 0.05 }}
                  layout
                >
                  {columns.map((column, colIndex) => (
                    <td
                      key={String(column.key)}
                      className={cn(
                        'font-mono text-sm border-b border-green-900/30',
                        column.align === 'center' && 'text-center',
                        column.align === 'right' && 'text-right',
                        rowIndex === data.length - 1 && 'border-b-0'
                      )}
                    >
                      {column.render
                        ? column.render(row[column.key], row, rowIndex)
                        : row[column.key]
                      }
                    </td>
                  ))}
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>
    </div>
  );
}