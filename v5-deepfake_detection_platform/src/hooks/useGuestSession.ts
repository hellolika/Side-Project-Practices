import { useState, useEffect } from 'react';
import { supabase } from '@/integrations/supabase/client';

interface GuestSession {
  sessionId: string;
  uploadCount: number;
  remainingUploads: number;
  expiresAt: string;
}

export const useGuestSession = () => {
  const [guestSession, setGuestSession] = useState<GuestSession | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    initializeGuestSession();
  }, []);

  const generateSessionId = () => {
    return 'guest_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  };

  const getClientInfo = () => {
    return {
      userAgent: navigator.userAgent,
      // Note: IP address would be handled server-side
    };
  };

  const initializeGuestSession = async () => {
    try {
      // Check if we have an existing session in localStorage
      const existingSessionId = localStorage.getItem('guest_session_id');
      
      if (existingSessionId) {
        // Verify the session is still valid
        const { data: session, error } = await supabase
          .from('guest_sessions_2025_11_27_05_00')
          .select('*')
          .eq('session_id', existingSessionId)
          .gt('expires_at', new Date().toISOString())
          .single();

        if (!error && session) {
          setGuestSession({
            sessionId: session.session_id,
            uploadCount: session.upload_count,
            remainingUploads: Math.max(0, 3 - session.upload_count),
            expiresAt: session.expires_at
          });
          setLoading(false);
          return;
        } else {
          // Session expired or invalid, remove from localStorage
          localStorage.removeItem('guest_session_id');
        }
      }

      // Create new guest session
      await createNewGuestSession();
    } catch (error) {
      console.error('Error initializing guest session:', error);
      setLoading(false);
    }
  };

  const createNewGuestSession = async () => {
    try {
      const sessionId = generateSessionId();
      const clientInfo = getClientInfo();

      const { data: session, error } = await supabase
        .from('guest_sessions_2025_11_27_05_00')
        .insert({
          session_id: sessionId,
          user_agent: clientInfo.userAgent,
          upload_count: 0
        })
        .select()
        .single();

      if (error) {
        throw error;
      }

      // Store session ID in localStorage
      localStorage.setItem('guest_session_id', sessionId);

      setGuestSession({
        sessionId: session.session_id,
        uploadCount: 0,
        remainingUploads: 3,
        expiresAt: session.expires_at
      });
    } catch (error) {
      console.error('Error creating guest session:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateUploadCount = (newCount: number) => {
    if (guestSession) {
      setGuestSession({
        ...guestSession,
        uploadCount: newCount,
        remainingUploads: Math.max(0, 3 - newCount)
      });
    }
  };

  const resetSession = async () => {
    localStorage.removeItem('guest_session_id');
    await createNewGuestSession();
  };

  const isLimitReached = () => {
    return guestSession ? guestSession.uploadCount >= 3 : false;
  };

  const getTimeRemaining = () => {
    if (!guestSession) return null;
    
    const expiresAt = new Date(guestSession.expiresAt);
    const now = new Date();
    const diff = expiresAt.getTime() - now.getTime();
    
    if (diff <= 0) return null;
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    return { hours, minutes };
  };

  return {
    guestSession,
    loading,
    updateUploadCount,
    resetSession,
    isLimitReached,
    getTimeRemaining,
    createNewGuestSession
  };
};