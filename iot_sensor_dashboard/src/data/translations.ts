import { Translation } from '@/types/language';

export const translations: Record<string, Translation> = {
  en: {
    // Header
    title: 'Industrial IoT Dashboard',
    subtitle: 'Real-time sensor monitoring and analytics',
    online: 'Online',
    offline: 'Offline',
    activeAlerts: 'Active Alerts',
    
    // Gauge labels
    temperature: 'Temperature',
    humidity: 'Humidity',
    pressure: 'Pressure',
    
    // Sensor grid
    sensorNetwork: 'Sensor Network',
    sensorGridDescription: 'Click on a sensor to view detailed information and historical trends',
    lastUpdate: 'Last update',
    temp: 'Temp',
    
    // Historical trend
    historicalTrend: 'Historical Trend',
    historicalDescription: 'Temperature and Pressure over the last 24 hours',
    noSensorSelected: 'No Sensor Selected',
    selectSensorMessage: 'Select a sensor from the grid to view historical data',
    currentTemperature: 'Current Temperature',
    currentHumidity: 'Current Humidity',
    currentPressure: 'Current Pressure',
    
    // Alert panel
    systemAlerts: 'System Alerts',
    active: 'Active',
    acknowledged: 'Acknowledged',
    noActiveAlerts: 'No active alerts',
    allSystemsNormal: 'All systems operating normally',
    acknowledge: 'Acknowledge',
    acknowledgedAlerts: 'Acknowledged Alerts',
    
    // Time units
    minutesAgo: 'm ago',
    hoursAgo: 'h ago',
    
    // Alert types
    critical: 'CRITICAL',
    warning: 'WARNING',
    info: 'INFO',
    
    // Units
    celsius: '°C',
    percent: '%',
    hpa: 'hPa',
  },
  
  km: {
    // Header
    title: 'ផ្ទាំងគ្រប់គ្រង IoT ឧស្សាហកម្ម',
    subtitle: 'ការត្រួតពិនិត្យ និងវិភាគទិន្នន័យពេលវេលាជាក់ស្តែង',
    online: 'អនឡាញ',
    offline: 'ក្រៅបណ្តាញ',
    activeAlerts: 'ការជូនដំណឹងសកម្ម',
    
    // Gauge labels
    temperature: 'សីតុណ្ហភាព',
    humidity: 'សំណើម',
    pressure: 'សម្ពាធ',
    
    // Sensor grid
    sensorNetwork: 'បណ្តាញឧបករណ៍ចាប់សញ្ញា',
    sensorGridDescription: 'ចុចលើឧបករណ៍ចាប់សញ្ញាដើម្បីមើលព័ត៌មានលម្អិត និងនិន្នាការប្រវត្តិសាស្ត្រ',
    lastUpdate: 'ការធ្វើបច្ចុប្បន្នភាពចុងក្រោយ',
    temp: 'សីតុណ្ហ',
    
    // Historical trend
    historicalTrend: 'និន្នាការប្រវត្តិសាស្ត្រ',
    historicalDescription: 'សីតុណ្ហភាព និងសម្ពាធក្នុងរយៈពេល ២៤ ម៉ោងចុងក្រោយ',
    noSensorSelected: 'មិនបានជ្រើសរើសឧបករណ៍ចាប់សញ្ញា',
    selectSensorMessage: 'ជ្រើសរើសឧបករណ៍ចាប់សញ្ញាពីក្រឡាចត្រង្គដើម្បីមើលទិន្នន័យប្រវត្តិសាស្ត្រ',
    currentTemperature: 'សីតុណ្ហភាពបច្ចុប្បន្ន',
    currentHumidity: 'សំណើមបច្ចុប្បន្ន',
    currentPressure: 'សម្ពាធបច្ចុប្បន្ន',
    
    // Alert panel
    systemAlerts: 'ការជូនដំណឹងប្រព័ន្ធ',
    active: 'សកម្ម',
    acknowledged: 'បានទទួលស្គាល់',
    noActiveAlerts: 'គ្មានការជូនដំណឹងសកម្ម',
    allSystemsNormal: 'ប្រព័ន្ធទាំងអស់ដំណើរការធម្មតា',
    acknowledge: 'ទទួលស្គាល់',
    acknowledgedAlerts: 'ការជូនដំណឹងដែលបានទទួលស្គាល់',
    
    // Time units
    minutesAgo: 'នាទីមុន',
    hoursAgo: 'ម៉ោងមុន',
    
    // Alert types
    critical: 'ធ្ងន់ធ្ងរ',
    warning: 'ការព្រមាន',
    info: 'ព័ត៌មាន',
    
    // Units
    celsius: '°C',
    percent: '%',
    hpa: 'hPa',
  },
};