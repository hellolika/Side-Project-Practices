import React, { useState } from 'react';
import { useAuth } from '@/components/Auth';
import { AuthForm } from '@/components/AuthForm';
import { GuestModePrompt } from '@/components/GuestModePrompt';
import { Header, StatsOverview } from '@/components/Header';
import { FileUpload } from '@/components/FileUpload';
import { AnalysisDashboard } from '@/components/AnalysisDashboard';
import { ForensicReports } from '@/components/ForensicReports';

const Index = () => {
  const { user, loading, isGuestMode } = useAuth();
  const [activeTab, setActiveTab] = useState('upload');

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  if (!user && !isGuestMode) {
    return <AuthForm />;
  }

  if (!user && isGuestMode) {
    // Guest mode - show limited interface
    return (
      <div className="min-h-screen bg-background">
        <Header activeTab={activeTab} onTabChange={setActiveTab} />
        <main className="container mx-auto px-4 py-6">
          <FileUpload />
        </main>
      </div>
    );
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'upload':
        return (
          <div className="space-y-6">
            <StatsOverview />
            <FileUpload />
          </div>
        );
      case 'dashboard':
        return <AnalysisDashboard />;
      case 'reports':
        return <ForensicReports />;
      default:
        return (
          <div className="space-y-6">
            <StatsOverview />
            <FileUpload />
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="container mx-auto px-4 py-6">
        {renderContent()}
      </main>
    </div>
  );
};

export default Index;
