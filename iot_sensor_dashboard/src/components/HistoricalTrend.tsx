import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Sensor } from '@/types/sensor';
import { useLanguage } from '@/contexts/LanguageContext';
import { useTheme } from '@/contexts/ThemeContext';

interface HistoricalTrendProps {
  sensor: Sensor | null;
}

export const HistoricalTrend: React.FC<HistoricalTrendProps> = ({ sensor }) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  if (!sensor) {
    return (
      <Card className="bg-slate-900 border-slate-700">
        <CardHeader>
          <CardTitle className="text-slate-100">{t('historicalTrend')} - 24 Hours</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-slate-400 text-center">
            <div className="text-lg mb-2">{t('noSensorSelected')}</div>
            <div className="text-sm">{t('selectSensorMessage')}</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for the chart (last 24 hours, sample every hour for readability)
  const chartData = sensor.historicalData
    .filter((_, index) => index % 12 === 0) // Sample every hour (every 12th point since we have 5-minute intervals)
    .map((reading) => ({
      time: reading.timestamp.toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit'
      }),
      temperature: Number(reading.temperature.toFixed(1)),
      pressure: Number(reading.pressure.toFixed(0)),
      humidity: Number(reading.humidity.toFixed(1)),
    }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-lg">
          <p className="text-slate-200 font-medium">{`Time: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.dataKey === 'temperature' ? 'Temperature' : 
                 entry.dataKey === 'pressure' ? 'Pressure' : 'Humidity'}: ${entry.value}${
                entry.dataKey === 'temperature' ? '°C' : 
                entry.dataKey === 'pressure' ? ' hPa' : '%'
              }`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-slate-900 border-slate-700">
      <CardHeader>
        <CardTitle className="text-slate-100">
          {t('historicalTrend')} - {sensor.name}
        </CardTitle>
        <div className="text-sm text-slate-400">
          {t('historicalDescription')}
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="time" 
                stroke="#9ca3af"
                fontSize={12}
                interval="preserveStartEnd"
              />
              <YAxis 
                yAxisId="temp"
                orientation="left"
                stroke="#ef4444"
                fontSize={12}
                domain={['dataMin - 5', 'dataMax + 5']}
              />
              <YAxis 
                yAxisId="pressure"
                orientation="right"
                stroke="#10b981"
                fontSize={12}
                domain={['dataMin - 20', 'dataMax + 20']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                wrapperStyle={{ color: '#e2e8f0' }}
              />
              <Line
                yAxisId="temp"
                type="monotone"
                dataKey="temperature"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                name="Temperature (°C)"
                activeDot={{ r: 4, fill: '#ef4444' }}
              />
              <Line
                yAxisId="pressure"
                type="monotone"
                dataKey="pressure"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="Pressure (hPa)"
                activeDot={{ r: 4, fill: '#10b981' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Current Values Summary */}
        <div className="mt-4 grid grid-cols-3 gap-4 pt-4 border-t border-slate-700">
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              {sensor.lastReading.temperature.toFixed(1)}°C
            </div>
            <div className="text-sm text-slate-400">{t('currentTemperature')}</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {sensor.lastReading.humidity.toFixed(1)}%
            </div>
            <div className="text-sm text-slate-400">{t('currentHumidity')}</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {sensor.lastReading.pressure.toFixed(0)} hPa
            </div>
            <div className="text-sm text-slate-400">{t('currentPressure')}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};