import React, { useState, useEffect } from 'react';
import { Calendar, ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { MatrixButton } from '../ui/MatrixButton';
import { MatrixInput } from '../ui/MatrixInput';
import { MatrixCard } from '../ui/MatrixCard';
import { format, subDays, subMonths, subYears, isAfter, isBefore, isSameDay, parseISO } from 'date-fns';

export interface DateRange {
  startDate: Date;
  endDate: Date;
}

interface DateRangePickerProps {
  value?: DateRange;
  onChange: (dateRange: DateRange) => void;
  placeholder?: string;
  minDate?: Date;
  maxDate?: Date;
  showTime?: boolean;
  disabled?: boolean;
  className?: string;
}

const DateRangePicker: React.FC<DateRangePickerProps> = ({
  value,
  onChange,
  placeholder = "Select date range",
  minDate,
  maxDate,
  showTime = false,
  disabled = false,
  className = ""
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [currentMonth, setCurrentMonth] = useState<Date>(value?.startDate || new Date());
  const [selectingStart, setSelectingStart] = useState(true);
  const [tempStartDate, setTempStartDate] = useState<Date | null>(value?.startDate || null);
  const [tempEndDate, setTempEndDate] = useState<Date | null>(value?.endDate || null);

  const presetRanges = [
    {
      label: 'Today',
      getValue: () => {
        const today = new Date();
        return { startDate: today, endDate: today };
      }
    },
    {
      label: 'Yesterday',
      getValue: () => {
        const yesterday = subDays(new Date(), 1);
        return { startDate: yesterday, endDate: yesterday };
      }
    },
    {
      label: 'Last 7 Days',
      getValue: () => {
        const end = new Date();
        const start = subDays(end, 6);
        return { startDate: start, endDate: end };
      }
    },
    {
      label: 'Last 30 Days',
      getValue: () => {
        const end = new Date();
        const start = subDays(end, 29);
        return { startDate: start, endDate: end };
      }
    },
    {
      label: 'Last 3 Months',
      getValue: () => {
        const end = new Date();
        const start = subMonths(end, 3);
        return { startDate: start, endDate: end };
      }
    },
    {
      label: 'Last 6 Months',
      getValue: () => {
        const end = new Date();
        const start = subMonths(end, 6);
        return { startDate: start, endDate: end };
      }
    },
    {
      label: 'Last Year',
      getValue: () => {
        const end = new Date();
        const start = subYears(end, 1);
        return { startDate: start, endDate: end };
      }
    },
    {
      label: 'Year to Date',
      getValue: () => {
        const end = new Date();
        const start = new Date(end.getFullYear(), 0, 1);
        return { startDate: start, endDate: end };
      }
    }
  ];

  const formatDateDisplay = (date: Date) => {
    if (!date) return '';
    const options: Intl.DateTimeFormatOptions = {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      ...(showTime && { hour: '2-digit', minute: '2-digit' })
    };
    return date.toLocaleDateString('en-US', options);
  };

  const formatDateForInput = (date: Date) => {
    if (!date) return '';
    return format(date, 'yyyy-MM-dd');
  };

  const handleApplyRange = () => {
    if (tempStartDate && tempEndDate && !isAfter(tempStartDate, tempEndDate)) {
      onChange({ startDate: tempStartDate, endDate: tempEndDate });
      setIsOpen(false);
    }
  };

  const handlePresetSelect = (preset: typeof presetRanges[0]) => {
    const range = preset.getValue();
    onChange(range);
    setIsOpen(false);
  };

  const handleDateClick = (date: Date) => {
    if (disabled) return;

    // Check if date is within allowed range
    if (minDate && isBefore(date, minDate)) return;
    if (maxDate && isAfter(date, maxDate)) return;

    if (selectingStart || !tempStartDate) {
      setTempStartDate(date);
      setTempEndDate(null);
      setSelectingStart(false);
    } else {
      if (isAfter(date, tempStartDate) || isSameDay(date, tempStartDate)) {
        setTempEndDate(date);
        setSelectingStart(true);
        // Auto-apply if end date is selected
        setTimeout(() => handleApplyRange(), 100);
      } else {
        setTempEndDate(date);
        setSelectingStart(true);
        setTimeout(() => handleApplyRange(), 100);
      }
    }
  };

  const isDateInRange = (date: Date) => {
    if (!tempStartDate || !tempEndDate) return false;
    return !isBefore(date, tempStartDate) && !isAfter(date, tempEndDate);
  };

  const isDateSelected = (date: Date) => {
    return (tempStartDate && isSameDay(date, tempStartDate)) ||
           (tempEndDate && isSameDay(date, tempEndDate));
  };

  const isDateDisabled = (date: Date) => {
    if (minDate && isBefore(date, minDate)) return true;
    if (maxDate && isAfter(date, maxDate)) return true;
    return false;
  };

  const generateCalendarDays = (month: Date) => {
    const year = month.getFullYear();
    const monthIndex = month.getMonth();
    
    const firstDay = new Date(year, monthIndex, 1);
    const lastDay = new Date(year, monthIndex + 1, 0);
    const startDate = new Date(firstDay);
    startDate.setDate(startDate.getDate() - firstDay.getDay());
    
    const days = [];
    const currentDate = new Date(startDate);
    
    for (let i = 0; i < 42; i++) {
      days.push(new Date(currentDate));
      currentDate.setDate(currentDate.getDate() + 1);
    }
    
    return days;
  };

  const calendarDays = generateCalendarDays(currentMonth);
  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  const displayValue = value ? 
    `${formatDateDisplay(value.startDate)} - ${formatDateDisplay(value.endDate)}` : 
    placeholder;

  return (
    <div className={`relative ${className}`}>
      <MatrixButton
        variant="outline"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className="w-full justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4" />
          <span className={value ? 'text-matrix-green' : 'text-matrix-green/70'}>
            {displayValue}
          </span>
        </div>
        <ChevronDown className={`w-4 h-4 transition-transform ${
          isOpen ? 'rotate-180' : ''
        }`} />
      </MatrixButton>

      {isOpen && (
        <MatrixCard className="absolute top-full left-0 right-0 z-50 mt-1 p-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Preset Ranges */}
            <div>
              <h4 className="text-matrix-green font-medium mb-2">Quick Select</h4>
              <div className="space-y-1">
                {presetRanges.map((preset) => (
                  <MatrixButton
                    key={preset.label}
                    variant="ghost"
                    onClick={() => handlePresetSelect(preset)}
                    className="w-full justify-start text-sm"
                  >
                    {preset.label}
                  </MatrixButton>
                ))}
              </div>
            </div>

            {/* Calendar */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <MatrixButton
                  variant="ghost"
                  size="sm"
                  onClick={() => setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1, 1))}
                >
                  <ChevronLeft className="w-4 h-4" />
                </MatrixButton>
                <h4 className="text-matrix-green font-medium">
                  {monthNames[currentMonth.getMonth()]} {currentMonth.getFullYear()}
                </h4>
                <MatrixButton
                  variant="ghost"
                  size="sm"
                  onClick={() => setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 1))}
                >
                  <ChevronRight className="w-4 h-4" />
                </MatrixButton>
              </div>

              {/* Day headers */}
              <div className="grid grid-cols-7 gap-1 mb-2">
                {dayNames.map(day => (
                  <div key={day} className="text-center text-xs text-matrix-green/70 py-1">
                    {day}
                  </div>
                ))}
              </div>

              {/* Calendar grid */}
              <div className="grid grid-cols-7 gap-1">
                {calendarDays.map((date, index) => {
                  const isCurrentMonth = date.getMonth() === currentMonth.getMonth();
                  const isInRange = isDateInRange(date);
                  const isSelected = isDateSelected(date);
                  const isDisabled = isDateDisabled(date);
                  
                  return (
                    <button
                      key={index}
                      onClick={() => handleDateClick(date)}
                      disabled={isDisabled}
                      className={`
                        p-1 text-sm rounded transition-colors
                        ${
                          isDisabled
                            ? 'text-matrix-green/30 cursor-not-allowed'
                            : 'text-matrix-green hover:bg-matrix-green/20 cursor-pointer'
                        }
                        ${
                          !isCurrentMonth ? 'opacity-30' : ''
                        }
                        ${
                          isSelected
                            ? 'bg-matrix-green text-matrix-darker font-medium'
                            : ''
                        }
                        ${
                          isInRange && !isSelected
                            ? 'bg-matrix-green/20'
                            : ''
                        }
                      `}
                    >
                      {date.getDate()}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Manual date inputs */}
          <div className="grid grid-cols-2 gap-2 mt-4 pt-4 border-t border-matrix-green/20">
            <div>
              <label className="block text-xs text-matrix-green/70 mb-1">Start Date</label>
              <MatrixInput
                type="date"
                value={tempStartDate ? formatDateForInput(tempStartDate) : ''}
                onChange={(e) => {
                  const date = e.target.value ? parseISO(e.target.value) : null;
                  setTempStartDate(date);
                  if (date && tempEndDate && isAfter(date, tempEndDate)) {
                    setTempEndDate(date);
                  }
                }}
                min={minDate ? formatDateForInput(minDate) : undefined}
                max={maxDate ? formatDateForInput(maxDate) : undefined}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-xs text-matrix-green/70 mb-1">End Date</label>
              <MatrixInput
                type="date"
                value={tempEndDate ? formatDateForInput(tempEndDate) : ''}
                onChange={(e) => {
                  const date = e.target.value ? parseISO(e.target.value) : null;
                  setTempEndDate(date);
                }}
                min={tempStartDate ? formatDateForInput(tempStartDate) : (minDate ? formatDateForInput(minDate) : undefined)}
                max={maxDate ? formatDateForInput(maxDate) : undefined}
                className="w-full"
              />
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2 mt-4 pt-4 border-t border-matrix-green/20">
            <MatrixButton
              variant="outline"
              onClick={() => {
                setTempStartDate(null);
                setTempEndDate(null);
                setSelectingStart(true);
              }}
              className="flex-1"
            >
              Clear
            </MatrixButton>
            <MatrixButton
              onClick={handleApplyRange}
              disabled={!tempStartDate || !tempEndDate}
              className="flex-1"
            >
              Apply
            </MatrixButton>
          </div>
        </MatrixCard>
      )}
    </div>
  );
};

export default DateRangePicker;