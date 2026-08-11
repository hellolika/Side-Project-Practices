import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useAuth } from './Auth';
import { useLanguage } from '@/contexts/LanguageContext';
import { ThemeSwitcher } from './ThemeSwitcher';
import { LanguageSwitcher } from './LanguageSwitcher';
import { 
  Shield, 
  Zap, 
  Eye, 
  FileText, 
  Users, 
  BarChart3,
  LogOut,
  Settings
} from 'lucide-react';

interface HeaderProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  const { user, signOut, isGuestMode, disableGuestMode } = useAuth();
  const { t } = useLanguage();

  return (
    <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="p-2 bg-primary rounded-lg">
                <Shield className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-bold">{t('header.title')}</h1>
                <p className="text-sm text-muted-foreground">{t('header.subtitle')}</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Theme and Language Switchers */}
            <ThemeSwitcher />
            <LanguageSwitcher />
            
            {isGuestMode ? (
              <>
                <div className="text-right">
                  <p className="text-sm font-medium text-orange-600 dark:text-orange-400">{t('header.demoMode')}</p>
                  <p className="text-xs text-muted-foreground">{t('header.guestUser')}</p>
                </div>
                <Button variant="outline" size="sm" onClick={disableGuestMode}>
                  <Users className="h-4 w-4 mr-2" />
                  {t('header.signUp')}
                </Button>
              </>
            ) : (
              <>
                <div className="text-right">
                  <p className="text-sm font-medium">{user?.email}</p>
                  <p className="text-xs text-muted-foreground">{t('header.analyst')}</p>
                </div>
                <Button variant="outline" size="sm" onClick={signOut}>
                  <LogOut className="h-4 w-4 mr-2" />
                  {t('header.signOut')}
                </Button>
              </>
            )}
          </div>
        </div>

        <nav className="flex gap-1 mt-4">
          <Button
            variant={activeTab === 'upload' ? 'default' : 'ghost'}
            onClick={() => onTabChange('upload')}
            className="flex items-center gap-2"
          >
            <Zap className="h-4 w-4" />
            {isGuestMode ? t('header.demoAnalysis') : t('header.uploadAnalyze')}
          </Button>
          {!isGuestMode && (
            <>
              <Button
                variant={activeTab === 'dashboard' ? 'default' : 'ghost'}
                onClick={() => onTabChange('dashboard')}
                className="flex items-center gap-2"
              >
                <BarChart3 className="h-4 w-4" />
                {t('header.analysisDashboard')}
              </Button>
              <Button
                variant={activeTab === 'reports' ? 'default' : 'ghost'}
                onClick={() => onTabChange('reports')}
                className="flex items-center gap-2"
              >
                <FileText className="h-4 w-4" />
                {t('header.forensicReports')}
              </Button>
            </>
          )}
        </nav>
      </div>
    </div>
  );
};

export const StatsOverview: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
      <Card className="forensic-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
              <Eye className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">1,247</p>
              <p className="text-sm text-muted-foreground">Files Analyzed</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="forensic-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg">
              <Shield className="h-5 w-5 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">89</p>
              <p className="text-sm text-muted-foreground">Deepfakes Detected</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="forensic-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-lg">
              <FileText className="h-5 w-5 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">156</p>
              <p className="text-sm text-muted-foreground">Reports Generated</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="forensic-card">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
              <BarChart3 className="h-5 w-5 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">97.3%</p>
              <p className="text-sm text-muted-foreground">Accuracy Rate</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};