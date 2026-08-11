export type Language = 'en' | 'km';

export interface Translation {
  // Header
  title: string;
  subtitle: string;
  online: string;
  warning: string;
  offline: string;
  activeAlerts: string;
  
  // Gauge labels
  temperature: string;
  humidity: string;
  pressure: string;
  
  // Sensor grid
  sensorNetwork: string;
  sensorGridDescription: string;
  lastUpdate: string;
  temp: string;
  
  // Historical trend
  historicalTrend: string;
  historicalDescription: string;
  noSensorSelected: string;
  selectSensorMessage: string;
  currentTemperature: string;
  currentHumidity: string;
  currentPressure: string;
  
  // Alert panel
  systemAlerts: string;
  active: string;
  acknowledged: string;
  noActiveAlerts: string;
  allSystemsNormal: string;
  acknowledge: string;
  acknowledgedAlerts: string;
  
  // Time units
  minutesAgo: string;
  hoursAgo: string;
  
  // Alert types
  critical: string;
  info: string;
  
  // Units
  celsius: string;
  percent: string;
  hpa: string;
}