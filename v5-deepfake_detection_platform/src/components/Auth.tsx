import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, Session } from '@supabase/supabase-js';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  isGuestMode: boolean;
  signUp: (email: string, password: string, userData?: { full_name?: string; organization?: string }) => Promise<{ data: any; error: any }>;
  signIn: (email: string, password: string) => Promise<{ data: any; error: any }>;
  signOut: () => Promise<void>;
  enableGuestMode: () => void;
  disableGuestMode: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [isGuestMode, setIsGuestMode] = useState(false);
  const { toast } = useToast();

  // Check for guest mode preference on load
  useEffect(() => {
    const guestModeEnabled = localStorage.getItem('guest_mode_enabled') === 'true';
    setIsGuestMode(guestModeEnabled);
  }, []);

  useEffect(() => {
    // Set up auth state listener
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        setLoading(false);
      }
    );

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signUp = async (email: string, password: string, userData?: { full_name?: string; organization?: string }) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: `${window.location.origin}/#/welcome`,
          data: userData
        }
      });

      if (error) throw error;

      toast({
        title: "Registration Successful",
        description: "Please check your email and click the confirmation link to complete registration",
      });

      return { data, error: null };
    } catch (error: any) {
      toast({
        title: "Registration Failed",
        description: error.message,
        variant: "destructive",
      });
      return { data: null, error };
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;

      toast({
        title: "Login Successful",
        description: "Welcome to the Deepfake Detection Platform",
      });

      return { data, error: null };
    } catch (error: any) {
      toast({
        title: "Login Failed",
        description: error.message,
        variant: "destructive",
      });
      return { data: null, error };
    }
  };

  const signOut = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;

      toast({
        title: "Logged Out",
        description: "You have been successfully logged out",
      });
    } catch (error: any) {
      toast({
        title: "Logout Failed",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  const enableGuestMode = () => {
    setIsGuestMode(true);
    localStorage.setItem('guest_mode_enabled', 'true');
    toast({
      title: "Demo Mode Enabled",
      description: "You can now test the platform with up to 3 file uploads",
    });
  };

  const disableGuestMode = () => {
    setIsGuestMode(false);
    localStorage.removeItem('guest_mode_enabled');
    localStorage.removeItem('guest_session_id');
  };

  const value = {
    user,
    session,
    loading,
    isGuestMode,
    signUp,
    signIn,
    signOut,
    enableGuestMode,
    disableGuestMode,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};