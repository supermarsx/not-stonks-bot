import React, { useState, useEffect } from 'react';
import { Calendar, ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { MatrixButton } from '../MatrixButton';
import { MatrixInput } from '../MatrixInput';
import { format, subDays, subMonths, subYears, isAfter, isBefore, parseISO } from 'date-fns';

interface DateRange {
  startDate: Date;
  endDate: Date;
}

interface DateRangePickerProps {
  onDateRangeChange: (dateRange: DateRange) => void;
  initialDateRange?: DateRange;
  presetRanges?: {
    label: string;
    getValue: () => { startDate: Date; endDate: Date };
  }[];
  className?: string;
  disabled?: boolean;
  maxDate?: Date;
  minDate?: Date;
}

export const DateRangePicker: React.FC<DateRangePickerProps> = ({
  onDateRangeChange,
  initialDateRange,
  presetRanges = [],
  className = "",
  disabled = false,
  maxDate = new Date(),
  minDate,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedRange, setSelectedRange] = useState<DateRange>(
    initialDateRange || {
      startDate: subMonths(new Date(), 3),
      endDate: new Date(),
    }
  );
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [view, setView] = useState<'range' | 'quick'>('quick');

  // Default preset ranges
  const defaultPresetRanges = [
    {
      label: 'Last 7 Days',
      getValue: () => ({
        startDate: subDays(new Date(), 7),
        endDate: new Date(),
      }),
    },
    {
      label: 'Last 30 Days',
      getValue: () => ({
        startDate: subDays(new Date(), 30),
        endDate: new Date(),
      }),
    },
    {
      label: 'Last 3 Months',
      getValue: () => ({
        startDate: subMonths(new Date(), 3),
        endDate: new Date(),
      }),
    },
    {
      label: 'Last 6 Months',
      getValue: () => ({
        startDate: subMonths(new Date(), 6),
        endDate: new Date(),
      }),
    },
    {
      label: 'Last Year',
      getValue: () => ({
        startDate: subYears(new Date(), 1),
        endDate: new Date(),
      }),
    },
    {
      label: 'Year to Date',
      getValue: () => ({
        startDate: new Date(new Date().getFullYear(), 0, 1),
        endDate: new Date(),
      }),
    },
  ];

  const allPresetRanges = [...defaultPresetRanges, ...presetRanges];

  // Calendar generation
  const getDaysInMonth = (date: Date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();

    const days = [];

    // Add empty cells for days before the first day of the month
    for (let i = 0; i < startingDayOfWeek; i++) {
      days.push(null);
    }

    // Add days of the month
    for (let day = 1; day <= daysInMonth; day++) {
      days.push(new Date(year, month, day));
    }

    return days;
  };

  const isDateInRange = (date: Date) => {
    return !isBefore(date, selectedRange.startDate) && !isAfter(date, selectedRange.endDate);
  };

  const isDateSelected = (date: Date) => {
    return date.toDateString() === selectedRange.startDate.toDateString() ||
           date.toDateString() === selectedRange.endDate.toDateString();
  };

  const handleDateClick = (date: Date | null) => {
    if (!date || disabled) return;

    if (view === 'range') {
      if (isBefore(date, selectedRange.startDate) || isAfter(date, selectedRange.endDate)) {
        // Reset range with new start date
        setSelectedRange({
          startDate: date,
          endDate: date,
        });
      } else {
        // Complete the range
        const newRange = {
          startDate: selectedRange.startDate,
          endDate: date,
        };
        setSelectedRange(newRange);
        onDateRangeChange(newRange);
        setIsOpen(false);
        setView('quick');
      }
    }
  };

  const handlePresetRangeSelect = (presetRange: typeof allPresetRanges[0]) => {
    const newRange = presetRange.getValue();
    setSelectedRange(newRange);
    onDateRangeChange(newRange);
    setIsOpen(false);
  };

  const navigateMonth = (direction: 'prev' | 'next') => {
    setCurrentMonth(prev => {
      const newMonth = new Date(prev);
      if (direction === 'prev') {
        newMonth.setMonth(prev.getMonth() - 1);
      } else {
        newMonth.setMonth(prev.getMonth() + 1);
      }
      return newMonth;
    });
  };

  const formatDisplayRange = () => {
    const formatOptions: Intl.DateTimeFormatOptions = { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    };
    return `${format(selectedRange.startDate, formatOptions)} - ${format(selectedRange.endDate, formatOptions)}`;
  };

  return (
    <div className={`relative ${className}`}>
      <MatrixButton
        variant="secondary"
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className="w-full justify-between min-w-[280px]"
      >
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4" />
          <span className="font-mono">{formatDisplayRange()}</span>
        </div>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </MatrixButton>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 bg-matrix-black border border-matrix-green rounded-lg shadow-lg z-50 p-4 min-w-[320px]">
          {/* View Toggle */}
          <div className="flex gap-2 mb-4">
            <MatrixButton
              size="sm"
              variant={view === 'quick' ? 'primary' : 'secondary'}
              onClick={() => setView('quick')}
            >
              Quick Ranges
            </MatrixButton>
            <MatrixButton
              size="sm"
              variant={view === 'range' ? 'primary' : 'secondary'}
              onClick={() => setView('range')}
            >
              Custom Range
            </MatrixButton>
          </div>

          {view === 'quick' ? (
            /* Quick Preset Ranges */
            <div className="space-y-2">
              <h4 className="text-matrix-green font-mono text-sm mb-3">Select Range</h4>
              {allPresetRanges.map((preset, index) => (
                <button
                  key={index}
                  onClick={() => handlePresetRangeSelect(preset)}
                  className="w-full text-left px-3 py-2 text-matrix-green hover:bg-matrix-green/10 rounded font-mono text-sm transition-colors"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          ) : (
            /* Custom Range Calendar */
            <div>
              <div className="flex items-center justify-between mb-4">
                <MatrixButton
                  size="sm"
                  variant="secondary"
                  onClick={() => navigateMonth('prev')}
                >
                  <ChevronLeft className="w-4 h-4" />
                </MatrixButton>
                <h4 className="text-matrix-green font-mono">
                  {format(currentMonth, 'MMMM yyyy')}
                </h4>
                <MatrixButton
                  size="sm"
                  variant="secondary"
                  onClick={() => navigateMonth('next')}
                  disabled={currentMonth >= maxDate}
                >
                  <ChevronRight className="w-4 h-4" />
                </MatrixButton>
              </div>

              <div className="grid grid-cols-7 gap-1 mb-2">
                {['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'].map(day => (
                  <div key={day} className="text-center text-matrix-green/70 text-xs font-mono py-1">
                    {day}
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-7 gap-1">
                {getDaysInMonth(currentMonth).map((date, index) => (
                  <button
                    key={index}
                    onClick={() => handleDateClick(date)}
                    disabled={!date || disabled || (minDate && isBefore(date, minDate)) || (maxDate && isAfter(date, maxDate))}
                    className={`
                      w-8 h-8 text-sm font-mono rounded transition-colors
                      ${!date ? '' : ''}
                      ${!date ? 'invisible' : ''}
                      ${date && isDateSelected(date) ? 'bg-matrix-green text-matrix-black' : ''}
                      ${date && isDateInRange(date) && !isDateSelected(date) ? 'bg-matrix-green/20 text-matrix-green' : ''}
                      ${date && !isDateInRange(date) ? 'text-matrix-green/50 hover:bg-matrix-green/10' : ''}
                      ${date && (minDate && isBefore(date, minDate) || maxDate && isAfter(date, maxDate)) ? 'text-matrix-green/30 cursor-not-allowed' : ''}
                      ${date && !isDateSelected(date) && !isDateInRange(date) ? 'hover:bg-matrix-green/10' : ''}
                    `}
                  >
                    {date?.getDate()}
                  </button>
                ))}
              </div>

              <div className="mt-4 pt-4 border-t border-matrix-green/30">
                <div className="flex gap-2">
                  <MatrixButton
                    size="sm"
                    variant="secondary"
                    onClick={() => setIsOpen(false)}
                  >
                    Cancel
                  </MatrixButton>
                  <MatrixButton
                    size="sm"
                    variant="primary"
                    onClick={() => {
                      onDateRangeChange(selectedRange);
                      setIsOpen(false);
                    }}
                  >
                    Apply
                  </MatrixButton>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DateRangePicker;