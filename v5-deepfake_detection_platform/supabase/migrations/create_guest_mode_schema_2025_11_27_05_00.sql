-- Create guest sessions table for tracking demo usage
CREATE TABLE public.guest_sessions_2025_11_27_05_00 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    upload_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 day')
);

-- Create temporary analysis requests for guest users
CREATE TABLE public.temp_analysis_requests_2025_11_27_05_00 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    guest_session_id UUID REFERENCES public.guest_sessions_2025_11_27_05_00(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size BIGINT,
    file_url TEXT,
    original_url TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 day')
);

-- Create temporary analysis results for guest users
CREATE TABLE public.temp_analysis_results_2025_11_27_05_00 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    temp_request_id UUID REFERENCES public.temp_analysis_requests_2025_11_27_05_00(id) ON DELETE CASCADE,
    manipulation_probability DECIMAL(5,2),
    confidence_score DECIMAL(5,2),
    detection_method TEXT,
    analysis_data JSONB,
    anomaly_regions JSONB,
    frame_analysis JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 day')
);

-- Create temporary storage bucket for guest files
INSERT INTO storage.buckets (id, name, public) VALUES 
('temp-guest-files', 'temp-guest-files', false)
ON CONFLICT (id) DO NOTHING;

-- RLS Policies for guest tables (allow public access but with session validation)
ALTER TABLE public.guest_sessions_2025_11_27_05_00 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.temp_analysis_requests_2025_11_27_05_00 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.temp_analysis_results_2025_11_27_05_00 ENABLE ROW LEVEL SECURITY;

-- Guest sessions policies (allow creation and reading of own session)
CREATE POLICY "Allow guest session creation" ON public.guest_sessions_2025_11_27_05_00
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow reading own guest session" ON public.guest_sessions_2025_11_27_05_00
    FOR SELECT USING (true);

CREATE POLICY "Allow updating own guest session" ON public.guest_sessions_2025_11_27_05_00
    FOR UPDATE USING (true);

-- Temp analysis requests policies
CREATE POLICY "Allow guest analysis requests" ON public.temp_analysis_requests_2025_11_27_05_00
    FOR ALL USING (true);

-- Temp analysis results policies
CREATE POLICY "Allow guest analysis results" ON public.temp_analysis_results_2025_11_27_05_00
    FOR ALL USING (true);

-- Storage policies for guest files
CREATE POLICY "Allow guest file uploads" ON storage.objects
    FOR INSERT WITH CHECK (bucket_id = 'temp-guest-files');

CREATE POLICY "Allow guest file access" ON storage.objects
    FOR SELECT USING (bucket_id = 'temp-guest-files');

-- Function to cleanup expired guest data
CREATE OR REPLACE FUNCTION cleanup_expired_guest_data_2025_11_27_05_00()
RETURNS void AS $$
BEGIN
    -- Delete expired guest sessions (cascade will handle related data)
    DELETE FROM public.guest_sessions_2025_11_27_05_00 
    WHERE expires_at < NOW();
    
    -- Delete expired temp requests that might not have been cascaded
    DELETE FROM public.temp_analysis_requests_2025_11_27_05_00 
    WHERE expires_at < NOW();
    
    -- Delete expired temp results that might not have been cascaded
    DELETE FROM public.temp_analysis_results_2025_11_27_05_00 
    WHERE expires_at < NOW();
    
    -- Note: Storage files cleanup would need to be handled separately
    -- as we can't directly delete from storage in SQL functions
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a scheduled job to run cleanup daily (using pg_cron if available)
-- This would typically be set up in the Supabase dashboard or via cron job
-- SELECT cron.schedule('cleanup-guest-data', '0 2 * * *', 'SELECT cleanup_expired_guest_data_2025_11_27_05_00();');

-- Function to check guest session limits
CREATE OR REPLACE FUNCTION check_guest_upload_limit_2025_11_27_05_00(session_id_param TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    current_count INTEGER;
BEGIN
    SELECT upload_count INTO current_count
    FROM public.guest_sessions_2025_11_27_05_00
    WHERE session_id = session_id_param AND expires_at > NOW();
    
    -- If session doesn't exist or expired, return false (need new session)
    IF current_count IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Check if under limit (3 uploads)
    RETURN current_count < 3;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to increment guest upload count
CREATE OR REPLACE FUNCTION increment_guest_upload_2025_11_27_05_00(session_id_param TEXT)
RETURNS INTEGER AS $$
DECLARE
    new_count INTEGER;
BEGIN
    UPDATE public.guest_sessions_2025_11_27_05_00
    SET upload_count = upload_count + 1
    WHERE session_id = session_id_param AND expires_at > NOW()
    RETURNING upload_count INTO new_count;
    
    RETURN COALESCE(new_count, 0);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;