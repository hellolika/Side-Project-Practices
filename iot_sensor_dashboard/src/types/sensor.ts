export interface SensorReading {
  timestamp: Date;
  temperature: number;
  humidity: number;
  pressure: number;
}

export interface Sensor {
  id: string;
  name: string;
  location: string;
  status: 'online' | 'offline' | 'warning';
  lastReading: SensorReading;
  historicalData: SensorReading[];
}

export interface Alert {
  id: string;
  sensorId: string;
  sensorName: string;
  type: 'critical' | 'warning' | 'info';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}