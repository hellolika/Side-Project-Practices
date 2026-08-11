import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useAuth } from './Auth';
import { useLanguage } from '@/contexts/LanguageContext';
import { Shield, Lock, Users } from 'lucide-react';

export const AuthForm: React.FC = () => {
  const { signIn, signUp, loading, enableGuestMode } = useAuth();
  const { t } = useLanguage();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fullName: '',
    organization: ''
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    await signIn(formData.email, formData.password);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    await signUp(formData.email, formData.password, {
      full_name: formData.fullName,
      organization: formData.organization
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="p-3 bg-primary rounded-full">
              <Shield className="h-8 w-8 text-primary-foreground" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {t('auth.title')}
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            {t('auth.subtitle')}
          </p>
        </div>

        <Card className="forensic-card">
          <CardHeader>
            <CardTitle className="text-center">{t('auth.secureAccess')}</CardTitle>
            <CardDescription className="text-center">
              {t('auth.governmentPortal')}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="signin" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="signin" className="flex items-center gap-2">
                  <Lock className="h-4 w-4" />
                  {t('auth.signIn')}
                </TabsTrigger>
                <TabsTrigger value="signup" className="flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  {t('auth.register')}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="signin">
                <form onSubmit={handleSignIn} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="signin-email">{t('auth.email')}</Label>
                    <Input
                      id="signin-email"
                      name="email"
                      type="email"
                      placeholder={t('auth.emailPlaceholder')}
                      value={formData.email}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signin-password">Password</Label>
                    <Input
                      id="signin-password"
                      name="password"
                      type="password"
                      value={formData.password}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? 'Signing In...' : 'Sign In'}
                  </Button>
                </form>
              </TabsContent>

              <TabsContent value="signup">
                <form onSubmit={handleSignUp} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="signup-name">Full Name</Label>
                    <Input
                      id="signup-name"
                      name="fullName"
                      type="text"
                      placeholder="Agent John Smith"
                      value={formData.fullName}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signup-organization">Organization</Label>
                    <Input
                      id="signup-organization"
                      name="organization"
                      type="text"
                      placeholder="FBI Digital Forensics Unit"
                      value={formData.organization}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signup-email">Email</Label>
                    <Input
                      id="signup-email"
                      name="email"
                      type="email"
                      placeholder="analyst@agency.gov"
                      value={formData.email}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="signup-password">Password</Label>
                    <Input
                      id="signup-password"
                      name="password"
                      type="password"
                      value={formData.password}
                      onChange={handleInputChange}
                      required
                    />
                  </div>
                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? 'Creating Account...' : 'Create Account'}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* Demo Mode Option */}
        <Card className="forensic-card mt-4 border-orange-200 dark:border-orange-800">
          <CardContent className="p-4">
            <div className="text-center space-y-3">
              <div className="flex items-center justify-center gap-2">
                <Shield className="h-5 w-5 text-orange-600 dark:text-orange-400" />
                <h3 className="font-medium text-orange-800 dark:text-orange-200">
                  {t('guest.tryDemo') || 'Try Demo Mode'}
                </h3>
              </div>
              <p className="text-sm text-orange-700 dark:text-orange-300">
                {t('guest.testWithoutReg') || 'Test our deepfake detection with 3 free uploads'}
              </p>
              <Button 
                onClick={enableGuestMode}
                variant="outline" 
                className="w-full border-orange-300 dark:border-orange-700 text-orange-700 dark:text-orange-300 hover:bg-orange-50 dark:hover:bg-orange-900/20"
              >
                <Shield className="h-4 w-4 mr-2" />
                {t('guest.startDemo') || 'Start Demo Mode'}
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="text-center mt-6 text-sm text-gray-600 dark:text-gray-400">
          <p>{t('auth.secureEncrypted')}</p>
        </div>
      </div>
    </div>
  );
};