import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAuth } from './Auth';
import { 
  Shield, 
  Zap, 
  Clock, 
  Upload,
  CheckCircle,
  AlertTriangle,
  Users,
  Lock
} from 'lucide-react';

export const GuestModePrompt: React.FC = () => {
  const { enableGuestMode } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="p-3 bg-primary rounded-full">
              <Shield className="h-8 w-8 text-primary-foreground" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Deepfake Detection Platform
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Professional forensics-grade media analysis
          </p>
        </div>

        {/* Demo Mode Card */}
        <Card className="forensic-card mb-6">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-3">
              <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-full">
                <Zap className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <CardTitle className="text-2xl">Try Our Demo Mode</CardTitle>
            <CardDescription className="text-lg">
              Test our deepfake detection technology without registration
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Demo Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-secondary rounded-lg">
                <Upload className="h-8 w-8 mx-auto mb-2 text-primary" />
                <h3 className="font-medium mb-1">3 Free Uploads</h3>
                <p className="text-sm text-muted-foreground">
                  Test with video, image, or audio files
                </p>
              </div>
              
              <div className="text-center p-4 bg-secondary rounded-lg">
                <Clock className="h-8 w-8 mx-auto mb-2 text-primary" />
                <h3 className="font-medium mb-1">24 Hour Access</h3>
                <p className="text-sm text-muted-foreground">
                  Full analysis results for one day
                </p>
              </div>
              
              <div className="text-center p-4 bg-secondary rounded-lg">
                <CheckCircle className="h-8 w-8 mx-auto mb-2 text-primary" />
                <h3 className="font-medium mb-1">Full Analysis</h3>
                <p className="text-sm text-muted-foreground">
                  Complete detection with confidence scores
                </p>
              </div>
            </div>

            {/* Demo Limitations */}
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                <div>
                  <h4 className="font-medium text-yellow-800 dark:text-yellow-200 mb-1">
                    Demo Mode Limitations
                  </h4>
                  <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
                    <li>• Limited to 3 file uploads per session</li>
                    <li>• Data automatically deleted after 24 hours</li>
                    <li>• No forensic report generation</li>
                    <li>• No data persistence or history</li>
                  </ul>
                </div>
              </div>
            </div>

            <Button onClick={enableGuestMode} size="lg" className="w-full">
              <Zap className="h-5 w-5 mr-2" />
              Start Demo Mode
            </Button>
          </CardContent>
        </Card>

        {/* Full Access Card */}
        <Card className="forensic-card">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-3">
              <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
                <Users className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
            <CardTitle className="text-xl">Need Full Access?</CardTitle>
            <CardDescription>
              Register for unlimited uploads and professional features
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Unlimited file uploads</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Permanent data storage</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Forensic report generation</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Batch processing</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Analysis history</span>
                </div>
                <div className="flex items-center gap-2">
                  <Lock className="h-4 w-4 text-blue-500" />
                  <span className="text-sm">Government-grade security</span>
                </div>
              </div>
            </div>

            <div className="text-center">
              <p className="text-sm text-muted-foreground mb-4">
                Professional forensics platform for government and law enforcement
              </p>
              <Badge variant="outline" className="mb-4">
                Secure • Encrypted • Forensics-Grade
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};