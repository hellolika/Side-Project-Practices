import { Sensor, SensorReading, Alert } from '@/types/sensor';

// Generate realistic sensor data
const generateSensorReading = (baseTemp: number = 22, baseHumidity: number = 45, basePressure: number = 1013): SensorReading => {
  return {
    timestamp: new Date(),
    temperature: baseTemp + (Math.random() - 0.5) * 10,
    humidity: Math.max(0, Math.min(100, baseHumidity + (Math.random() - 0.5) * 20)),
    pressure: basePressure + (Math.random() - 0.5) * 50,
  };
};

// Generate historical data for the last 24 hours
const generateHistoricalData = (sensorId: string): SensorReading[] => {
  const data: SensorReading[] = [];
  const now = new Date();
  const baseTemp = 20 + Math.random() * 10;
  const baseHumidity = 40 + Math.random() * 20;
  const basePressure = 1000 + Math.random() * 50;

  for (let i = 24 * 60; i >= 0; i -= 5) { // Every 5 minutes for 24 hours
    const timestamp = new Date(now.getTime() - i * 60 * 1000);
    const timeOfDay = timestamp.getHours();
    
    // Simulate daily temperature variation
    const tempVariation = Math.sin((timeOfDay - 6) * Math.PI / 12) * 5;
    const temperature = baseTemp + tempVariation + (Math.random() - 0.5) * 2;
    
    const humidity = Math.max(0, Math.min(100, baseHumidity + (Math.random() - 0.5) * 10));
    const pressure = basePressure + (Math.random() - 0.5) * 20;

    data.push({
      timestamp,
      temperature,
      humidity,
      pressure,
    });
  }

  return data;
};

// Mock sensor data
export const mockSensors: Sensor[] = [
  {
    id: 'sensor-001',
    name: 'Production Line A',
    location: 'Factory Floor - Zone 1',
    status: 'online',
    lastReading: generateSensorReading(25, 50, 1015),
    historicalData: generateHistoricalData('sensor-001'),
  },
  {
    id: 'sensor-002',
    name: 'Storage Area B',
    location: 'Warehouse - Section 2',
    status: 'warning',
    lastReading: generateSensorReading(35, 75, 1008), // High temp and humidity
    historicalData: generateHistoricalData('sensor-002'),
  },
  {
    id: 'sensor-003',
    name: 'Clean Room C',
    location: 'Laboratory - Room 301',
    status: 'online',
    lastReading: generateSensorReading(22, 40, 1020),
    historicalData: generateHistoricalData('sensor-003'),
  },
  {
    id: 'sensor-004',
    name: 'Cooling System D',
    location: 'HVAC - Unit 4',
    status: 'offline',
    lastReading: generateSensorReading(18, 30, 1012),
    historicalData: generateHistoricalData('sensor-004'),
  },
  {
    id: 'sensor-005',
    name: 'Boiler Room E',
    location: 'Basement - Utility',
    status: 'online',
    lastReading: generateSensorReading(45, 60, 1005),
    historicalData: generateHistoricalData('sensor-005'),
  },
  {
    id: 'sensor-006',
    name: 'Server Room F',
    location: 'IT Department - Floor 3',
    status: 'warning',
    lastReading: generateSensorReading(28, 45, 1018),
    historicalData: generateHistoricalData('sensor-006'),
  },
];

// Mock alerts
export const mockAlerts: Alert[] = [
  {
    id: 'alert-001',
    sensorId: 'sensor-002',
    sensorName: 'Storage Area B',
    type: 'critical',
    message: 'Temperature exceeds safe operating range (35°C)',
    timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
    acknowledged: false,
  },
  {
    id: 'alert-002',
    sensorId: 'sensor-004',
    sensorName: 'Cooling System D',
    type: 'critical',
    message: 'Sensor offline - No data received for 2 hours',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    acknowledged: false,
  },
  {
    id: 'alert-003',
    sensorId: 'sensor-006',
    sensorName: 'Server Room F',
    type: 'warning',
    message: 'Humidity levels approaching upper threshold (45%)',
    timestamp: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
    acknowledged: true,
  },
  {
    id: 'alert-004',
    sensorId: 'sensor-002',
    sensorName: 'Storage Area B',
    type: 'warning',
    message: 'Humidity exceeds recommended levels (75%)',
    timestamp: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
    acknowledged: false,
  },
];

// Utility function to update sensor readings (for real-time simulation)
export const updateSensorReading = (sensor: Sensor): Sensor => {
  const newReading = generateSensorReading(
    sensor.lastReading.temperature,
    sensor.lastReading.humidity,
    sensor.lastReading.pressure
  );

  // Determine status based on readings
  let status: 'online' | 'offline' | 'warning' = 'online';
  
  if (Math.random() < 0.05) { // 5% chance of going offline
    status = 'offline';
  } else if (
    newReading.temperature > 30 || 
    newReading.temperature < 15 ||
    newReading.humidity > 70 ||
    newReading.pressure < 1000 ||
    newReading.pressure > 1030
  ) {
    status = 'warning';
  }

  return {
    ...sensor,
    status,
    lastReading: newReading,
    historicalData: [...sensor.historicalData.slice(1), newReading], // Keep last 24 hours
  };
};