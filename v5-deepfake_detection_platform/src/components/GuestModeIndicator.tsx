import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useGuestSession } from '@/hooks/useGuestSession';
import { useAuth } from './Auth';
import { 
  Clock, 
  Upload, 
  AlertTriangle, 
  Users,
  Zap
} from 'lucide-react';

export const GuestModeIndicator: React.FC = () => {
  const { guestSession, isLimitReached, getTimeRemaining } = useGuestSession();
  const { disableGuestMode } = useAuth();

  if (!guestSession) return null;

  const timeRemaining = getTimeRemaining();
  const progressPercentage = (guestSession.uploadCount / 3) * 100;

  return (
    <Card className="mb-6 border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/20">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-orange-600 dark:text-orange-400" />
            <span className="font-medium text-orange-800 dark:text-orange-200">
              Demo Mode Active
            </span>
            <Badge variant="outline" className="text-orange-700 dark:text-orange-300">
              Guest Session
            </Badge>
          </div>
          
          <Button 
            variant="outline" 
            size="sm" 
            onClick={disableGuestMode}
            className="text-orange-700 dark:text-orange-300 border-orange-300 dark:border-orange-700"
          >
            <Users className="h-4 w-4 mr-1" />
            Sign Up for Full Access
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Upload Progress */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-orange-800 dark:text-orange-200">
                Uploads Used
              </span>
              <span className="text-sm text-orange-600 dark:text-orange-400">
                {guestSession.uploadCount} / 3
              </span>
            </div>
            <Progress 
              value={progressPercentage} 
              className="h-2"
            />
            {isLimitReached() && (
              <div className="flex items-center gap-1 text-sm text-red-600 dark:text-red-400">
                <AlertTriangle className="h-4 w-4" />
                <span>Upload limit reached</span>
              </div>
            )}
          </div>

          {/* Time Remaining */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-orange-600 dark:text-orange-400" />
              <span className="text-sm font-medium text-orange-800 dark:text-orange-200">
                Session Expires
              </span>
            </div>
            {timeRemaining ? (
              <p className="text-sm text-orange-600 dark:text-orange-400">
                {timeRemaining.hours}h {timeRemaining.minutes}m remaining
              </p>
            ) : (
              <p className="text-sm text-red-600 dark:text-red-400">
                Session expired
              </p>
            )}
          </div>

          {/* Limitations */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-orange-600 dark:text-orange-400" />
              <span className="text-sm font-medium text-orange-800 dark:text-orange-200">
                Demo Limitations
              </span>
            </div>
            <ul className="text-xs text-orange-600 dark:text-orange-400 space-y-1">
              <li>• No data persistence</li>
              <li>• No forensic reports</li>
              <li>• 24-hour auto-cleanup</li>
            </ul>
          </div>
        </div>

        {isLimitReached() && (
          <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
              <span className="text-sm font-medium text-red-800 dark:text-red-200">
                Upload Limit Reached
              </span>
            </div>
            <p className="text-sm text-red-700 dark:text-red-300 mb-3">
              You've used all 3 demo uploads. Register for unlimited access to continue analyzing files.
            </p>
            <Button 
              size="sm" 
              onClick={disableGuestMode}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <Users className="h-4 w-4 mr-1" />
              Create Account Now
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};