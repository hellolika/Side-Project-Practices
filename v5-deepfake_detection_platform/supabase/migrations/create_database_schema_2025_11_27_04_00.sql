-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create profiles table for user management
CREATE TABLE public.profiles_2025_11_27_04_00 (
    id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
    email TEXT,
    full_name TEXT,
    organization TEXT,
    role TEXT DEFAULT 'analyst',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create analysis_requests table for tracking file analysis
CREATE TABLE public.analysis_requests_2025_11_27_04_00 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL, -- 'video', 'image', 'audio'
    file_size BIGINT,
    file_url TEXT,
    original_url TEXT, -- for URL-based analysis
    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    batch_id UUID, -- for batch processing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create analysis_results table for storing detection results
CREATE TABLE public.analysis_results_2025_11_27_04_00 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    request_id UUID REFERENCES public.analysis_requests_2025_11_27_04_00(id) ON DELETE CASCADE,
    manipulation_probability DECIMAL(5,2), -- percentage 0.00-100.00
    confidence_score DECIMAL(5,2),
    detection_method TEXT,
    analysis_data JSONB, -- store heatmap data, spectrogram data, etc.
    anomaly_regions JSONB, -- coordinates of suspicious regions
    frame_analysis JSONB, -- for video frame-by-frame analysis
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create forensic_reports table for PDF reports
CREATE TABLE public.forensic_reports_2025_11_27_04_00 (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    request_id UUID REFERENCES public.analysis_requests_2025_11_27_04_00(id) ON DELETE CASCADE,
    report_url TEXT,
    report_data JSONB,
    generated_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create storage buckets
INSERT INTO storage.buckets (id, name, public) VALUES 
('media-files', 'media-files', false),
('forensic-reports', 'forensic-reports', false);

-- RLS Policies
ALTER TABLE public.profiles_2025_11_27_04_00 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_requests_2025_11_27_04_00 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results_2025_11_27_04_00 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.forensic_reports_2025_11_27_04_00 ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile" ON public.profiles_2025_11_27_04_00
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles_2025_11_27_04_00
    FOR UPDATE USING (auth.uid() = id);

-- Analysis requests policies
CREATE POLICY "Users can view own requests" ON public.analysis_requests_2025_11_27_04_00
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create requests" ON public.analysis_requests_2025_11_27_04_00
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Analysis results policies
CREATE POLICY "Users can view results for own requests" ON public.analysis_results_2025_11_27_04_00
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.analysis_requests_2025_11_27_04_00 
            WHERE id = request_id AND user_id = auth.uid()
        )
    );

-- Forensic reports policies
CREATE POLICY "Users can view reports for own requests" ON public.forensic_reports_2025_11_27_04_00
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.analysis_requests_2025_11_27_04_00 
            WHERE id = request_id AND user_id = auth.uid()
        )
    );

-- Storage policies
CREATE POLICY "Users can upload media files" ON storage.objects
    FOR INSERT WITH CHECK (bucket_id = 'media-files' AND auth.role() = 'authenticated');

CREATE POLICY "Users can view own media files" ON storage.objects
    FOR SELECT USING (bucket_id = 'media-files' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can view own reports" ON storage.objects
    FOR SELECT USING (bucket_id = 'forensic-reports' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Function to create user profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user_2025_11_27_04_00()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles_2025_11_27_04_00 (id, email, full_name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger for new user creation
CREATE TRIGGER on_auth_user_created_2025_11_27_04_00
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user_2025_11_27_04_00();