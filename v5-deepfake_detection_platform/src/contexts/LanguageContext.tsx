import React, { createContext, useContext, useState, useEffect } from 'react';

type Language = 'en' | 'km';

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

// Translation dictionaries
const translations = {
  en: {
    // Header
    'header.title': 'Deepfake Detection Platform',
    'header.subtitle': 'Professional Forensics Suite',
    'header.demoMode': 'Demo Mode',
    'header.guestUser': 'Guest User',
    'header.signUp': 'Sign Up',
    'header.signOut': 'Sign Out',
    'header.analyst': 'Digital Forensics Analyst',
    'header.demoAnalysis': 'Demo Analysis',
    'header.uploadAnalyze': 'Upload & Analyze',
    'header.analysisDashboard': 'Analysis Dashboard',
    'header.forensicReports': 'Forensic Reports',

    // Authentication
    'auth.title': 'Deepfake Detection Platform',
    'auth.subtitle': 'Professional forensics-grade media analysis',
    'auth.secureAccess': 'Secure Access',
    'auth.governmentPortal': 'Government and law enforcement portal',
    'auth.signIn': 'Sign In',
    'auth.register': 'Register',
    'auth.email': 'Email',
    'auth.password': 'Password',
    'auth.fullName': 'Full Name',
    'auth.organization': 'Organization',
    'auth.emailPlaceholder': 'analyst@agency.gov',
    'auth.namePlaceholder': 'Agent John Smith',
    'auth.orgPlaceholder': 'FBI Digital Forensics Unit',
    'auth.signingIn': 'Signing In...',
    'auth.creatingAccount': 'Creating Account...',
    'auth.createAccount': 'Create Account',
    'auth.secureEncrypted': 'Secure • Encrypted • Forensics-Grade',

    // Guest Mode
    'guest.tryDemo': 'Try Our Demo Mode',
    'guest.testWithoutReg': 'Test our deepfake detection technology without registration',
    'guest.freeUploads': '3 Free Uploads',
    'guest.testFiles': 'Test with video, image, or audio files',
    'guest.hourAccess': '24 Hour Access',
    'guest.fullResults': 'Full analysis results for one day',
    'guest.fullAnalysis': 'Full Analysis',
    'guest.completeDetection': 'Complete detection with confidence scores',
    'guest.limitations': 'Demo Mode Limitations',
    'guest.limitedUploads': '• Limited to 3 file uploads per session',
    'guest.autoDelete': '• Data automatically deleted after 24 hours',
    'guest.noReports': '• No forensic report generation',
    'guest.noPersistence': '• No data persistence or history',
    'guest.startDemo': 'Start Demo Mode',
    'guest.needFullAccess': 'Need Full Access?',
    'guest.registerUnlimited': 'Register for unlimited uploads and professional features',
    'guest.unlimitedUploads': 'Unlimited file uploads',
    'guest.permanentStorage': 'Permanent data storage',
    'guest.reportGeneration': 'Forensic report generation',
    'guest.batchProcessing': 'Batch processing',
    'guest.analysisHistory': 'Analysis history',
    'guest.governmentSecurity': 'Government-grade security',
    'guest.professionalPlatform': 'Professional forensics platform for government and law enforcement',

    // File Upload
    'upload.title': 'Media Upload & Analysis',
    'upload.description': 'Upload video, image, or audio files for deepfake detection analysis',
    'upload.fileUpload': 'File Upload',
    'upload.urlAnalysis': 'URL Analysis',
    'upload.dropFiles': 'Drop files here or click to upload',
    'upload.supportedFormats': 'Supports: MP4, MOV (video) • JPG, PNG (image) • WAV, MP3 (audio)',
    'upload.uploadedFiles': 'Uploaded Files:',
    'upload.remove': 'Remove',
    'upload.mediaUrl': 'Media URL',
    'upload.urlPlaceholder': 'https://example.com/video.mp4',
    'upload.urlDescription': 'Enter a direct URL to a video, image, or audio file for analysis',
    'upload.processing': 'Processing files...',
    'upload.complete': '% complete',
    'upload.startAnalysis': 'Start Analysis',
    'upload.limitReached': 'Demo Limit Reached',

    // Analysis Dashboard
    'dashboard.title': 'Analysis Dashboard',
    'dashboard.description': 'Review deepfake detection results and generate reports',
    'dashboard.refresh': 'Refresh',
    'dashboard.noResults': 'No Analysis Results',
    'dashboard.uploadToSee': 'Upload and analyze media files to see results here',
    'dashboard.manipulationRisk': 'Manipulation Risk:',
    'dashboard.probability': 'Probability:',
    'dashboard.confidence': 'Confidence:',
    'dashboard.method': 'Method:',
    'dashboard.details': 'Details',
    'dashboard.report': 'Report',
    'dashboard.analyzing': 'Analyzing...',
    'dashboard.pendingAnalysis': 'Pending Analysis',
    'dashboard.detailedAnalysis': 'Detailed Analysis:',
    'dashboard.close': 'Close',

    // Risk Levels
    'risk.high': 'HIGH',
    'risk.medium': 'MEDIUM',
    'risk.low': 'LOW',
    'risk.highRisk': 'HIGH RISK',
    'risk.mediumRisk': 'MEDIUM RISK',
    'risk.lowRisk': 'LOW RISK',

    // Welcome Page
    'welcome.title': 'Welcome to the Deepfake Detection Platform',
    'welcome.verified': 'Your account has been successfully verified!',
    'welcome.confirmed': 'Email confirmed • Account activated',
    'welcome.advancedAI': 'Advanced AI Detection',
    'welcome.aiDescription': 'State-of-the-art algorithms for detecting face-swaps, synthetic media, and manipulated content',
    'welcome.professionalAnalysis': 'Professional Analysis',
    'welcome.analysisDescription': 'Comprehensive forensic analysis with detailed reports and visualizations for legal proceedings',
    'welcome.secureCompliant': 'Secure & Compliant',
    'welcome.securityDescription': 'Government-grade security with chain of custody documentation and audit trails',
    'welcome.gettingStarted': 'Getting Started',
    'welcome.followSteps': 'Follow these steps to begin your forensic analysis',
    'welcome.uploadMedia': 'Upload Media Files',
    'welcome.uploadDescription': 'Upload video, image, or audio files for analysis. Supports batch processing for multiple files.',
    'welcome.reviewAnalysis': 'Review Analysis',
    'welcome.reviewDescription': 'Monitor processing status and review detailed analysis results with confidence scores.',
    'welcome.generateReports': 'Generate Reports',
    'welcome.reportsDescription': 'Create professional forensic reports with technical analysis and legal documentation.',
    'welcome.startAnalyzing': 'Start Analyzing Media',

    // Common
    'common.loading': 'Loading...',
    'common.error': 'Error',
    'common.success': 'Success',
    'common.cancel': 'Cancel',
    'common.save': 'Save',
    'common.delete': 'Delete',
    'common.edit': 'Edit',
    'common.view': 'View',
    'common.download': 'Download',
    'common.upload': 'Upload',
    'common.processing': 'Processing',
    'common.completed': 'Completed',
    'common.failed': 'Failed',
    'common.pending': 'Pending',
  },
  km: {
    // Header
    'header.title': 'វេទិកាការរកឃើញ Deepfake',
    'header.subtitle': 'ឈុតឧបករណ៍វិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',
    'header.demoMode': 'របៀបសាកល្បង',
    'header.guestUser': 'អ្នកប្រើភ្ញៀវ',
    'header.signUp': 'ចុះឈ្មោះ',
    'header.signOut': 'ចាកចេញ',
    'header.analyst': 'អ្នកវិភាគវិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',
    'header.demoAnalysis': 'ការវិភាគសាកល្បង',
    'header.uploadAnalyze': 'ផ្ទុកឡើង និងវិភាគ',
    'header.analysisDashboard': 'ផ្ទាំងគ្រប់គ្រងការវិភាគ',
    'header.forensicReports': 'របាយការណ៍វិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',

    // Authentication
    'auth.title': 'វេទិកាការរកឃើញ Deepfake',
    'auth.subtitle': 'ការវិភាគមេឌៀកម្រិតវិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',
    'auth.secureAccess': 'ការចូលប្រើប្រាស់មានសុវត្ថិភាព',
    'auth.governmentPortal': 'ច្រកចូលរដ្ឋាភិបាល និងអនុវត្តច្បាប់',
    'auth.signIn': 'ចូល',
    'auth.register': 'ចុះឈ្មោះ',
    'auth.email': 'អ៊ីមែល',
    'auth.password': 'ពាក្យសម្ងាត់',
    'auth.fullName': 'ឈ្មោះពេញ',
    'auth.organization': 'អង្គការ',
    'auth.emailPlaceholder': 'analyst@agency.gov',
    'auth.namePlaceholder': 'ភ្នាក់ងារ ជន ស្មីត',
    'auth.orgPlaceholder': 'អង្គភាពវិទ្យាសាស្ត្រព្រហ្មទណ្ឌ FBI',
    'auth.signingIn': 'កំពុងចូល...',
    'auth.creatingAccount': 'កំពុងបង្កើតគណនី...',
    'auth.createAccount': 'បង្កើតគណនី',
    'auth.secureEncrypted': 'មានសុវត្ថិភាព • បានអ៊ិនគ្រីប • កម្រិតវិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',

    // Guest Mode
    'guest.tryDemo': 'សាកល្បងរបៀបសាកល្បងរបស់យើង',
    'guest.testWithoutReg': 'សាកល្បងបច្ចេកវិទ្យារកឃើញ deepfake ដោយមិនចាំបាច់ចុះឈ្មោះ',
    'guest.freeUploads': 'ផ្ទុកឡើងឥតគិតថ្លៃ ៣ ដង',
    'guest.testFiles': 'សាកល្បងជាមួយឯកសារវីដេអូ រូបភាព ឬសំឡេង',
    'guest.hourAccess': 'ការចូលប្រើ ២៤ ម៉ោង',
    'guest.fullResults': 'លទ្ធផលការវិភាគពេញលេញសម្រាប់មួយថ្ងៃ',
    'guest.fullAnalysis': 'ការវិភាគពេញលេញ',
    'guest.completeDetection': 'ការរកឃើញពេញលេញជាមួយពិន្ទុទំនុកចិត្ត',
    'guest.limitations': 'ការកំណត់របៀបសាកល្បង',
    'guest.limitedUploads': '• កំណត់ត្រឹម ៣ ដងក្នុងមួយសម័យ',
    'guest.autoDelete': '• ទិន្នន័យត្រូវបានលុបដោយស្វ័យប្រវត្តិបន្ទាប់ពី ២៤ ម៉ោង',
    'guest.noReports': '• មិនមានការបង្កើតរបាយការណ៍វិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',
    'guest.noPersistence': '• មិនមានការរក្សាទុកទិន្នន័យ ឬប្រវត្តិ',
    'guest.startDemo': 'ចាប់ផ្តើមរបៀបសាកល្បង',
    'guest.needFullAccess': 'ត្រូវការការចូលប្រើពេញលេញ?',
    'guest.registerUnlimited': 'ចុះឈ្មោះសម្រាប់ការផ្ទុកឡើងគ្មានកំណត់ និងលក្ខណៈពិសេសវិជ្ជាជីវៈ',
    'guest.unlimitedUploads': 'ការផ្ទុកឡើងឯកសារគ្មានកំណត់',
    'guest.permanentStorage': 'ការរក្សាទុកទិន្នន័យអចិន្ត្រៃយ៍',
    'guest.reportGeneration': 'ការបង្កើតរបាយការណ៍វិទ្យាសាស្ត្រព្រហ្មទណ្ឌ',
    'guest.batchProcessing': 'ការដំណើរការជាបាច់',
    'guest.analysisHistory': 'ប្រវត្តិការវិភាគ',
    'guest.governmentSecurity': 'សុវត្ថិភាពកម្រិតរដ្ឋាភិបាល',
    'guest.professionalPlatform': 'វេទិកាវិទ្យាសាស្ត្រព្រហ្មទណ្ឌវិជ្ជាជីវៈសម្រាប់រដ្ឋាភិបាល និងអនុវត្តច្បាប់',

    // File Upload
    'upload.title': 'ការផ្ទុកឡើង និងការវិភាគមេឌៀ',
    'upload.description': 'ផ្ទុកឡើងឯកសារវីដេអូ រូបភាព ឬសំឡេងសម្រាប់ការវិភាគការរកឃើញ deepfake',
    'upload.fileUpload': 'ការផ្ទុកឡើងឯកសារ',
    'upload.urlAnalysis': 'ការវិភាគ URL',
    'upload.dropFiles': 'ទម្លាក់ឯកសារនៅទីនេះ ឬចុចដើម្បីផ្ទុកឡើង',
    'upload.supportedFormats': 'គាំទ្រ: MP4, MOV (វីដេអូ) • JPG, PNG (រូបភាព) • WAV, MP3 (សំឡេង)',
    'upload.uploadedFiles': 'ឯកសារដែលបានផ្ទុកឡើង:',
    'upload.remove': 'យកចេញ',
    'upload.mediaUrl': 'URL មេឌៀ',
    'upload.urlPlaceholder': 'https://example.com/video.mp4',
    'upload.urlDescription': 'បញ្ចូល URL ផ្ទាល់ទៅកាន់ឯកសារវីដេអូ រូបភាព ឬសំឡេងសម្រាប់ការវិភាគ',
    'upload.processing': 'កំពុងដំណើរការឯកសារ...',
    'upload.complete': '% បានបញ្ចប់',
    'upload.startAnalysis': 'ចាប់ផ្តើមការវិភាគ',
    'upload.limitReached': 'បានដល់កំណត់សាកល្បង',

    // Analysis Dashboard
    'dashboard.title': 'ផ្ទាំងគ្រប់គ្រងការវិភាគ',
    'dashboard.description': 'ពិនិត្យលទ្ធផលការរកឃើញ deepfake និងបង្កើតរបាយការណ៍',
    'dashboard.refresh': 'ធ្វើឱ្យស្រស់',
    'dashboard.noResults': 'គ្មានលទ្ធផលការវិភាគ',
    'dashboard.uploadToSee': 'ផ្ទុកឡើង និងវិភាគឯកសារមេឌៀដើម្បីមើលលទ្ធផលនៅទីនេះ',
    'dashboard.manipulationRisk': 'ហានិភ័យនៃការរៀបចំ:',
    'dashboard.probability': 'ប្រូបាប៊ីលីតេ:',
    'dashboard.confidence': 'ទំនុកចិត្ត:',
    'dashboard.method': 'វិធីសាស្ត្រ:',
    'dashboard.details': 'លម្អិត',
    'dashboard.report': 'របាយការណ៍',
    'dashboard.analyzing': 'កំពុងវិភាគ...',
    'dashboard.pendingAnalysis': 'ការវិភាគកំពុងរង់ចាំ',
    'dashboard.detailedAnalysis': 'ការវិភាគលម្អិត:',
    'dashboard.close': 'បិទ',

    // Risk Levels
    'risk.high': 'ខ្ពស់',
    'risk.medium': 'មធ្យម',
    'risk.low': 'ទាប',
    'risk.highRisk': 'ហានិភ័យខ្ពស់',
    'risk.mediumRisk': 'ហានិភ័យមធ្យម',
    'risk.lowRisk': 'ហានិភ័យទាប',

    // Welcome Page
    'welcome.title': 'សូមស្វាគមន៍មកកាន់វេទិកាការរកឃើញ Deepfake',
    'welcome.verified': 'គណនីរបស់អ្នកត្រូវបានផ្ទៀងផ្ទាត់ដោយជោគជ័យ!',
    'welcome.confirmed': 'អ៊ីមែលបានបញ្ជាក់ • គណនីបានធ្វើឱ្យសកម្ម',
    'welcome.advancedAI': 'ការរកឃើញ AI កម្រិតខ្ពស់',
    'welcome.aiDescription': 'ក្បួនដោះស្រាយទំនើបបំផុតសម្រាប់ការរកឃើញការផ្លាស់ប្តូរមុខ មេឌៀសំយោគ និងមាតិកាដែលបានរៀបចំ',
    'welcome.professionalAnalysis': 'ការវិភាគវិជ្ជាជីវៈ',
    'welcome.analysisDescription': 'ការវិភាគវិទ្យាសាស្ត្រព្រហ្មទណ្ឌគ្រប់ជ្រុងជ្រោយជាមួយរបាយការណ៍លម្អិត និងការមើលឃើញសម្រាប់ដំណើរការតុលាការ',
    'welcome.secureCompliant': 'មានសុវត្ថិភាព និងអនុលោម',
    'welcome.securityDescription': 'សុវត្ថិភាពកម្រិតរដ្ឋាភិបាលជាមួយឯកសារខ្សែសង្វាក់ការយាមកាម និងដាននៃការសវនកម្ម',
    'welcome.gettingStarted': 'ការចាប់ផ្តើម',
    'welcome.followSteps': 'អនុវត្តតាមជំហានទាំងនេះដើម្បីចាប់ផ្តើមការវិភាគវិទ្យាសាស្ត្រព្រហ្មទណ្ឌរបស់អ្នក',
    'welcome.uploadMedia': 'ផ្ទុកឡើងឯកសារមេឌៀ',
    'welcome.uploadDescription': 'ផ្ទុកឡើងឯកសារវីដេអូ រូបភាព ឬសំឡេងសម្រាប់ការវិភាគ។ គាំទ្រការដំណើរការជាបាច់សម្រាប់ឯកសារច្រើន។',
    'welcome.reviewAnalysis': 'ពិនិត្យការវិភាគ',
    'welcome.reviewDescription': 'តាមដានស្ថានភាពដំណើរការ និងពិនិត្យលទ្ធផលការវិភាគលម្អិតជាមួយពិន្ទុទំនុកចិត្ត។',
    'welcome.generateReports': 'បង្កើតរបាយការណ៍',
    'welcome.reportsDescription': 'បង្កើតរបាយការណ៍វិទ្យាសាស្ត្រព្រហ្មទណ្ឌវិជ្ជាជីវៈជាមួយការវិភាគបច្ចេកទេស និងឯកសារច្បាប់។',
    'welcome.startAnalyzing': 'ចាប់ផ្តើមវិភាគមេឌៀ',

    // Common
    'common.loading': 'កំពុងផ្ទុក...',
    'common.error': 'កំហុស',
    'common.success': 'ជោគជ័យ',
    'common.cancel': 'បោះបង់',
    'common.save': 'រក្សាទុក',
    'common.delete': 'លុប',
    'common.edit': 'កែសម្រួល',
    'common.view': 'មើល',
    'common.download': 'ទាញយក',
    'common.upload': 'ផ្ទុកឡើង',
    'common.processing': 'កំពុងដំណើរការ',
    'common.completed': 'បានបញ្ចប់',
    'common.failed': 'បរាជ័យ',
    'common.pending': 'កំពុងរង់ចាំ',
  }
};

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguage] = useState<Language>('en');

  useEffect(() => {
    // Load language from localStorage
    const savedLanguage = localStorage.getItem('language') as Language;
    if (savedLanguage && ['en', 'km'].includes(savedLanguage)) {
      setLanguage(savedLanguage);
    }
  }, []);

  useEffect(() => {
    // Save to localStorage
    localStorage.setItem('language', language);
  }, [language]);

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  const value = {
    language,
    setLanguage,
    t,
  };

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
};