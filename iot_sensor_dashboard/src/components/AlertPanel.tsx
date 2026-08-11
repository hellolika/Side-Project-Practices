import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert } from '@/types/sensor';
import { AlertTriangle, AlertCircle, Info, Clock, CheckCircle } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';
import { useTheme } from '@/contexts/ThemeContext';

interface AlertPanelProps {
  alerts: Alert[];
  onAcknowledge: (alertId: string) => void;
}

export const AlertPanel: React.FC<AlertPanelProps> = ({ alerts, onAcknowledge }) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical':
        return <AlertTriangle className="w-4 h-4" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4" />;
      case 'info':
        return <Info className="w-4 h-4" />;
      default:
        return <Info className="w-4 h-4" />;
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'critical':
        return 'text-red-400 bg-red-900/20 border-red-800';
      case 'warning':
        return 'text-yellow-400 bg-yellow-900/20 border-yellow-800';
      case 'info':
        return 'text-blue-400 bg-blue-900/20 border-blue-800';
      default:
        return 'text-gray-400 bg-gray-900/20 border-gray-800';
    }
  };

  const getBadgeVariant = (type: string) => {
    switch (type) {
      case 'critical':
        return 'destructive';
      case 'warning':
        return 'secondary';
      case 'info':
        return 'default';
      default:
        return 'outline';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 60) {
      return `${diffMins}${t('minutesAgo')}`;
    } else if (diffHours < 24) {
      return `${diffHours}${t('hoursAgo')}`;
    } else {
      return timestamp.toLocaleDateString();
    }
  };

  const activeAlerts = alerts.filter(alert => !alert.acknowledged);
  const acknowledgedAlerts = alerts.filter(alert => alert.acknowledged);

  return (
    <Card className="bg-slate-900 border-slate-700">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-slate-100">{t('systemAlerts')}</CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="destructive" className="text-xs">
              {activeAlerts.length} {t('active')}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {acknowledgedAlerts.length} {t('acknowledged')}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
            <div className="text-slate-400">{t('noActiveAlerts')}</div>
            <div className="text-sm text-slate-500">{t('allSystemsNormal')}</div>
          </div>
        ) : (
          <>
            {/* Active Alerts */}
            {activeAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-lg border ${getAlertColor(alert.type)} transition-all duration-200`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1">
                    <div className="mt-0.5">
                      {getAlertIcon(alert.type)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={getBadgeVariant(alert.type)} className="text-xs">
                          {t(alert.type).toUpperCase()}
                        </Badge>
                        <span className="text-sm font-medium text-slate-200">
                          {alert.sensorName}
                        </span>
                      </div>
                      <div className="text-sm text-slate-300 mb-2">
                        {alert.message}
                      </div>
                      <div className="flex items-center text-xs text-slate-500">
                        <Clock className="w-3 h-3 mr-1" />
                        {formatTimestamp(alert.timestamp)}
                      </div>
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onAcknowledge(alert.id)}
                    className="ml-2 text-xs"
                  >
                    {t('acknowledge')}
                  </Button>
                </div>
              </div>
            ))}

            {/* Acknowledged Alerts */}
            {acknowledgedAlerts.length > 0 && (
              <>
                <div className="border-t border-slate-700 pt-3 mt-4">
                  <div className="text-sm text-slate-400 mb-3 font-medium">
                    {t('acknowledgedAlerts')}
                  </div>
                </div>
                {acknowledgedAlerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="p-3 rounded-lg border border-slate-700 bg-slate-800/50 opacity-60"
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5 text-slate-500">
                        <CheckCircle className="w-4 h-4" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline" className="text-xs">
                            {t(alert.type).toUpperCase()}
                          </Badge>
                          <span className="text-sm font-medium text-slate-400">
                            {alert.sensorName}
                          </span>
                        </div>
                        <div className="text-sm text-slate-500 mb-2">
                          {alert.message}
                        </div>
                        <div className="flex items-center text-xs text-slate-600">
                          <Clock className="w-3 h-3 mr-1" />
                          {formatTimestamp(alert.timestamp)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};