import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useLanguage } from '@/contexts/LanguageContext';
import { useTheme } from '@/contexts/ThemeContext';

interface GaugeProps {
  title: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  thresholds?: {
    warning: number;
    critical: number;
  };
}

export const Gauge: React.FC<GaugeProps> = ({ 
  title, 
  value, 
  unit, 
  min, 
  max, 
  thresholds 
}) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  const percentage = Math.min(Math.max((value - min) / (max - min) * 100, 0), 100);
  const angle = (percentage / 100) * 180; // Half circle
  
  // Determine color based on thresholds
  let color = 'text-green-500';
  let strokeColor = '#10b981'; // green-500
  
  if (thresholds) {
    if (value >= thresholds.critical) {
      color = 'text-red-500';
      strokeColor = '#ef4444'; // red-500
    } else if (value >= thresholds.warning) {
      color = 'text-yellow-500';
      strokeColor = '#eab308'; // yellow-500
    }
  }

  const radius = 80;
  const strokeWidth = 8;
  const normalizedRadius = radius - strokeWidth * 2;
  const circumference = normalizedRadius * Math.PI; // Half circle
  const strokeDasharray = `${circumference} ${circumference}`;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <Card className={`transition-colors duration-200 ${
      theme === 'light' 
        ? 'bg-white border-slate-200' 
        : 'bg-slate-900 border-slate-700'
    }`}>
      <CardHeader className="pb-2">
        <CardTitle className={`text-sm font-medium ${
          theme === 'light' ? 'text-slate-700' : 'text-slate-300'
        }`}>{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center">
        <div className="relative w-40 h-20 mb-4">
          <svg
            className="w-full h-full transform -rotate-90"
            viewBox={`0 0 ${radius * 2} ${radius}`}
          >
            {/* Background arc */}
            <path
              d={`M ${strokeWidth} ${radius - strokeWidth} A ${normalizedRadius} ${normalizedRadius} 0 0 1 ${radius * 2 - strokeWidth} ${radius - strokeWidth}`}
              fill="none"
              stroke={theme === 'light' ? '#e2e8f0' : '#374151'}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
            />
            {/* Progress arc */}
            <path
              d={`M ${strokeWidth} ${radius - strokeWidth} A ${normalizedRadius} ${normalizedRadius} 0 0 1 ${radius * 2 - strokeWidth} ${radius - strokeWidth}`}
              fill="none"
              stroke={strokeColor}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              className="transition-all duration-500 ease-in-out"
            />
          </svg>
          {/* Needle */}
          <div 
            className="absolute top-full left-1/2 w-0.5 h-16 bg-slate-400 origin-bottom transform -translate-x-1/2 transition-transform duration-500"
            style={{ transform: `translateX(-50%) rotate(${angle - 90}deg)` }}
          />
          {/* Center dot */}
          <div className="absolute top-full left-1/2 w-3 h-3 bg-slate-400 rounded-full transform -translate-x-1/2 -translate-y-1.5" />
        </div>
        
        <div className="text-center">
          <div className={`text-2xl font-bold ${color}`}>
            {value.toFixed(1)}
          </div>
          <div className={`text-sm ${
            theme === 'light' ? 'text-slate-600' : 'text-slate-400'
          }`}>{unit}</div>
          <div className={`text-xs mt-1 ${
            theme === 'light' ? 'text-slate-500' : 'text-slate-500'
          }`}>
            {min} - {max} {unit}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};