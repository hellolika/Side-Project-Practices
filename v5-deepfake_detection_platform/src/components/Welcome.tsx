import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/components/Auth';
import { useNavigate } from 'react-router-dom';
import { 
  Shield, 
  CheckCircle, 
  Zap, 
  FileText, 
  BarChart3,
  ArrowRight,
  Users,
  Lock,
  Eye
} from 'lucide-react';

export const Welcome: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [showAnimation, setShowAnimation] = useState(false);

  useEffect(() => {
    // Trigger animation after component mounts
    const timer = setTimeout(() => setShowAnimation(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const handleGetStarted = () => {
    navigate('/');
  };

  // Redirect to main app if user is not authenticated
  useEffect(() => {
    if (!user) {
      navigate('/');
    }
  }, [user, navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        {/* Welcome Header */}
        <div className={`text-center mb-8 transition-all duration-1000 ${showAnimation ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-primary rounded-full shadow-lg">
              <Shield className="h-12 w-12 text-primary-foreground" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Welcome to the Deepfake Detection Platform
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-2">
            Your account has been successfully verified!
          </p>
          <div className="flex items-center justify-center gap-2 text-green-600 dark:text-green-400">
            <CheckCircle className="h-5 w-5" />
            <span className="font-medium">Email confirmed • Account activated</span>
          </div>
        </div>

        {/* Feature Cards */}
        <div className={`grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 transition-all duration-1000 delay-300 ${showAnimation ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          <Card className="forensic-card hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <div className="flex justify-center mb-3">
                <div className="p-3 bg-blue-100 dark:bg-blue-900/20 rounded-full">
                  <Zap className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
              </div>
              <CardTitle className="text-lg">Advanced AI Detection</CardTitle>
              <CardDescription>
                State-of-the-art algorithms for detecting face-swaps, synthetic media, and manipulated content
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="forensic-card hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <div className="flex justify-center mb-3">
                <div className="p-3 bg-green-100 dark:bg-green-900/20 rounded-full">
                  <BarChart3 className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
              </div>
              <CardTitle className="text-lg">Professional Analysis</CardTitle>
              <CardDescription>
                Comprehensive forensic analysis with detailed reports and visualizations for legal proceedings
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="forensic-card hover:shadow-lg transition-shadow">
            <CardHeader className="text-center">
              <div className="flex justify-center mb-3">
                <div className="p-3 bg-purple-100 dark:bg-purple-900/20 rounded-full">
                  <Lock className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                </div>
              </div>
              <CardTitle className="text-lg">Secure & Compliant</CardTitle>
              <CardDescription>
                Government-grade security with chain of custody documentation and audit trails
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* Getting Started Section */}
        <Card className={`forensic-card mb-8 transition-all duration-1000 delay-500 ${showAnimation ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          <CardHeader>
            <CardTitle className="text-center text-2xl">Getting Started</CardTitle>
            <CardDescription className="text-center">
              Follow these steps to begin your forensic analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center space-y-3">
                <div className="flex justify-center">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold text-lg">
                    1
                  </div>
                </div>
                <h3 className="font-medium">Upload Media Files</h3>
                <p className="text-sm text-muted-foreground">
                  Upload video, image, or audio files for analysis. Supports batch processing for multiple files.
                </p>
              </div>

              <div className="text-center space-y-3">
                <div className="flex justify-center">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold text-lg">
                    2
                  </div>
                </div>
                <h3 className="font-medium">Review Analysis</h3>
                <p className="text-sm text-muted-foreground">
                  Monitor processing status and review detailed analysis results with confidence scores.
                </p>
              </div>

              <div className="text-center space-y-3">
                <div className="flex justify-center">
                  <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center text-primary-foreground font-bold text-lg">
                    3
                  </div>
                </div>
                <h3 className="font-medium">Generate Reports</h3>
                <p className="text-sm text-muted-foreground">
                  Create professional forensic reports with technical analysis and legal documentation.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* User Info & Action */}
        <div className={`text-center transition-all duration-1000 delay-700 ${showAnimation ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          <Card className="forensic-card inline-block">
            <CardContent className="p-6">
              <div className="flex items-center gap-4 mb-4">
                <div className="p-2 bg-secondary rounded-full">
                  <Users className="h-5 w-5" />
                </div>
                <div className="text-left">
                  <p className="font-medium">Welcome, {user?.email}</p>
                  <p className="text-sm text-muted-foreground">Digital Forensics Analyst</p>
                </div>
              </div>
              
              <Button onClick={handleGetStarted} className="w-full" size="lg">
                <Eye className="h-5 w-5 mr-2" />
                Start Analyzing Media
                <ArrowRight className="h-5 w-5 ml-2" />
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Security Notice */}
        <div className={`text-center mt-8 transition-all duration-1000 delay-900 ${showAnimation ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Shield className="h-4 w-4" />
            <span>Secure • Encrypted • Forensics-Grade • Government Compliant</span>
          </div>
        </div>
      </div>
    </div>
  );
};