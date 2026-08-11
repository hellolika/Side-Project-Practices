import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sensor } from '@/types/sensor';
import { Thermometer, Droplets, Gauge as GaugeIcon, MapPin, Clock } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';
import { useTheme } from '@/contexts/ThemeContext';

interface SensorGridProps {
  sensors: Sensor[];
  selectedSensorId?: string;
  onSensorSelect: (sensorId: string) => void;
}

export const SensorGrid: React.FC<SensorGridProps> = ({ 
  sensors, 
  selectedSensorId, 
  onSensorSelect 
}) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'bg-green-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'offline':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'online':
        return 'default';
      case 'warning':
        return 'secondary';
      case 'offline':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {sensors.map((sensor) => (
        <Card
          key={sensor.id}
          className={`cursor-pointer transition-all duration-200 hover:shadow-lg ${
            theme === 'light'
              ? 'bg-white border-slate-200 hover:border-slate-300'
              : 'bg-slate-900 border-slate-700 hover:border-slate-600'
          } ${
            selectedSensorId === sensor.id ? 'ring-2 ring-blue-500 border-blue-500' : ''
          }`}
          onClick={() => onSensorSelect(sensor.id)}
        >
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className={`text-lg font-semibold ${
                theme === 'light' ? 'text-slate-900' : 'text-slate-100'
              }`}>
                {sensor.name}
              </CardTitle>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${getStatusColor(sensor.status)} animate-pulse`} />
                <Badge variant={getStatusBadgeVariant(sensor.status)} className="text-xs">
                  {t(sensor.status).toUpperCase()}
                </Badge>
              </div>
            </div>
            <div className={`flex items-center text-sm ${
              theme === 'light' ? 'text-slate-600' : 'text-slate-400'
            }`}>
              <MapPin className="w-4 h-4 mr-1" />
              {sensor.location}
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4">
            {/* Key Readings */}
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <div className="flex items-center justify-center mb-1">
                  <Thermometer className="w-4 h-4 text-red-400" />
                </div>
                <div className={`text-lg font-semibold ${
                  theme === 'light' ? 'text-slate-900' : 'text-slate-100'
                }`}>
                  {sensor.lastReading.temperature.toFixed(1)}°
                </div>
                <div className={`text-xs ${
                  theme === 'light' ? 'text-slate-600' : 'text-slate-400'
                }`}>{t('temp')}</div>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center mb-1">
                  <Droplets className="w-4 h-4 text-blue-400" />
                </div>
                <div className={`text-lg font-semibold ${
                  theme === 'light' ? 'text-slate-900' : 'text-slate-100'
                }`}>
                  {sensor.lastReading.humidity.toFixed(1)}%
                </div>
                <div className={`text-xs ${
                  theme === 'light' ? 'text-slate-600' : 'text-slate-400'
                }`}>{t('humidity')}</div>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center mb-1">
                  <GaugeIcon className="w-4 h-4 text-green-400" />
                </div>
                <div className={`text-lg font-semibold ${
                  theme === 'light' ? 'text-slate-900' : 'text-slate-100'
                }`}>
                  {sensor.lastReading.pressure.toFixed(0)}
                </div>
                <div className={`text-xs ${
                  theme === 'light' ? 'text-slate-600' : 'text-slate-400'
                }`}>hPa</div>
              </div>
            </div>

            {/* Last Update */}
            <div className={`flex items-center justify-center text-xs pt-2 border-t ${
              theme === 'light' 
                ? 'text-slate-500 border-slate-200' 
                : 'text-slate-500 border-slate-700'
            }`}>
              <Clock className="w-3 h-3 mr-1" />
              {t('lastUpdate')}: {formatTimestamp(sensor.lastReading.timestamp)}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};