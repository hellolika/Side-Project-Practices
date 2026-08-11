import React, { useState, useEffect } from 'react';
import { Gauge } from '@/components/Gauge';
import { SensorGrid } from '@/components/SensorGrid';
import { HistoricalTrend } from '@/components/HistoricalTrend';
import { AlertPanel } from '@/components/AlertPanel';
import { LanguageSelector } from '@/components/LanguageSelector';
import { ThemeToggle } from '@/components/ThemeToggle';
import { mockSensors, mockAlerts, updateSensorReading } from '@/data/mockData';
import { Sensor, Alert } from '@/types/sensor';
import { Activity, Wifi, WifiOff, AlertTriangle } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';
import { useTheme } from '@/contexts/ThemeContext';

const Index = () => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [sensors, setSensors] = useState<Sensor[]>(mockSensors);
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);
  const [selectedSensorId, setSelectedSensorId] = useState<string | null>(null);

  const selectedSensor = selectedSensorId 
    ? sensors.find(s => s.id === selectedSensorId) || null 
    : null;

  // Real-time data simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setSensors(prevSensors => 
        prevSensors.map(sensor => updateSensorReading(sensor))
      );
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  // Auto-select first sensor on load
  useEffect(() => {
    if (sensors.length > 0 && !selectedSensorId) {
      setSelectedSensorId(sensors[0].id);
    }
  }, [sensors, selectedSensorId]);

  const handleSensorSelect = (sensorId: string) => {
    setSelectedSensorId(sensorId);
  };

  const handleAcknowledgeAlert = (alertId: string) => {
    setAlerts(prevAlerts => 
      prevAlerts.map(alert => 
        alert.id === alertId 
          ? { ...alert, acknowledged: true }
          : alert
      )
    );
  };

  const onlineSensors = sensors.filter(s => s.status === 'online').length;
  const warningSensors = sensors.filter(s => s.status === 'warning').length;
  const offlineSensors = sensors.filter(s => s.status === 'offline').length;
  const activeAlerts = alerts.filter(a => !a.acknowledged).length;

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      theme === 'light' 
        ? 'bg-slate-50 text-slate-900' 
        : 'bg-slate-950 text-slate-100'
    }`}>
      {/* Header */}
      <div className={`border-b px-6 py-6 relative transition-colors duration-200 ${
        theme === 'light'
          ? 'bg-white border-slate-200'
          : 'bg-slate-900 border-slate-700'
      }`}>
        {/* Controls */}
        <div className="absolute top-3 right-3 sm:right-6 z-20 flex items-center gap-2 sm:gap-3">
          <ThemeToggle />
          <LanguageSelector />
        </div>
        
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mt-2 gap-4 sm:gap-0">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-400" />
            <div>
              <h1 className={`text-xl sm:text-2xl font-bold ${
                theme === 'light' ? 'text-slate-900' : 'text-slate-100'
              }`}>
                {t('title')}
              </h1>
              <p className={`text-sm ${
                theme === 'light' ? 'text-slate-600' : 'text-slate-400'
              }`}>
                {t('subtitle')}
              </p>
            </div>
          </div>
          
          {/* System Status */}
          <div className="flex flex-wrap items-center gap-2 sm:gap-3 lg:gap-6 sm:mr-0">
            <div className="flex items-center gap-2">
              <Wifi className="w-5 h-5 text-green-400" />
              <span className={`text-sm ${
                theme === 'light' ? 'text-slate-700' : 'text-slate-300'
              }`}>
                {onlineSensors} {t('online')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
              <span className={`text-sm ${
                theme === 'light' ? 'text-slate-700' : 'text-slate-300'
              }`}>
                {warningSensors} {t('warning')}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <WifiOff className="w-5 h-5 text-red-400" />
              <span className={`text-sm ${
                theme === 'light' ? 'text-slate-700' : 'text-slate-300'
              }`}>
                {offlineSensors} {t('offline')}
              </span>
            </div>
            {activeAlerts > 0 && (
              <div className="flex items-center gap-2 px-3 py-1 bg-red-900/30 border border-red-800 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-300">
                  {activeAlerts} {t('activeAlerts')}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Gauge Components Row */}
        {selectedSensor && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Gauge
              title={t('temperature')}
              value={selectedSensor.lastReading.temperature}
              unit="°C"
              min={-10}
              max={50}
              thresholds={{ warning: 30, critical: 40 }}
            />
            <Gauge
              title={t('humidity')}
              value={selectedSensor.lastReading.humidity}
              unit="%"
              min={0}
              max={100}
              thresholds={{ warning: 70, critical: 85 }}
            />
            <Gauge
              title={t('pressure')}
              value={selectedSensor.lastReading.pressure}
              unit="hPa"
              min={950}
              max={1050}
              thresholds={{ warning: 1000, critical: 980 }}
            />
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sensor Grid - Takes 2 columns */}
          <div className="lg:col-span-2">
            <div className="mb-4">
              <h2 className={`text-xl font-semibold mb-2 ${
                theme === 'light' ? 'text-slate-900' : 'text-slate-100'
              }`}>
                {t('sensorNetwork')}
              </h2>
              <p className={`text-sm ${
                theme === 'light' ? 'text-slate-600' : 'text-slate-400'
              }`}>
                {t('sensorGridDescription')}
              </p>
            </div>
            <SensorGrid
              sensors={sensors}
              selectedSensorId={selectedSensorId || undefined}
              onSensorSelect={handleSensorSelect}
            />
          </div>

          {/* Alert Panel - Takes 1 column */}
          <div>
            <AlertPanel
              alerts={alerts}
              onAcknowledge={handleAcknowledgeAlert}
            />
          </div>
        </div>

        {/* Historical Trend Chart - Full Width */}
        <div>
          <HistoricalTrend sensor={selectedSensor} />
        </div>
      </div>
    </div>
  );
};

export default Index;
