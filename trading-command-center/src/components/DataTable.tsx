import { ReactNode } from 'react';
import { cn } from '../utils/cn';

interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (item: T) => ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  keyField: keyof T;
  onRowClick?: (item: T) => void;
  loading?: boolean;
  emptyMessage?: string;
}

export function DataTable<T>({
  data,
  columns,
  keyField,
  onRowClick,
  loading = false,
  emptyMessage = 'No data available',
}: DataTableProps<T>) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-matrix-green animate-pulse">LOADING DATA...</div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-matrix-green/50">{emptyMessage}</div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b-2 border-matrix-green">
            {columns.map((column, index) => (
              <th
                key={index}
                className={cn(
                  'px-4 py-3 text-left text-sm font-bold text-matrix-green',
                  column.className
                )}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((item, rowIndex) => (
            <tr
              key={String(item[keyField])}
              onClick={() => onRowClick?.(item)}
              className={cn(
                'border-b border-matrix-dark-green transition-colors',
                onRowClick && 'cursor-pointer hover:bg-matrix-dark-green/30'
              )}
            >
              {columns.map((column, colIndex) => (
                <td
                  key={colIndex}
                  className={cn(
                    'px-4 py-3 text-sm text-matrix-green/90',
                    column.className
                  )}
                >
                  {column.render
                    ? column.render(item)
                    : String(item[column.key as keyof T] ?? '-')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
