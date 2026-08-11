import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Authorization, X-Client-Info, apikey, Content-Type, X-Application-Name, X-Guest-Session-Id',
};

interface GuestAnalysisRequest {
  fileName: string;
  fileType: string;
  fileSize: number;
  fileUrl?: string;
  originalUrl?: string;
  guestSessionId: string;
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    const { fileName, fileType, fileSize, fileUrl, originalUrl, guestSessionId }: GuestAnalysisRequest = await req.json();

    if (!guestSessionId) {
      throw new Error('Guest session ID is required');
    }

    // Check if guest session exists and is within limits
    const { data: canUpload, error: limitError } = await supabaseClient
      .rpc('check_guest_upload_limit_2025_11_27_05_00', { session_id_param: guestSessionId });

    if (limitError) {
      throw new Error(`Session validation failed: ${limitError.message}`);
    }

    if (!canUpload) {
      return new Response(
        JSON.stringify({ 
          error: 'Upload limit exceeded or session expired',
          code: 'LIMIT_EXCEEDED',
          message: 'You have reached the 3-file limit for demo mode. Please register for unlimited access.'
        }),
        { 
          status: 429,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      );
    }

    // Get guest session details
    const { data: guestSession, error: sessionError } = await supabaseClient
      .from('guest_sessions_2025_11_27_05_00')
      .select('*')
      .eq('session_id', guestSessionId)
      .single();

    if (sessionError || !guestSession) {
      throw new Error('Invalid guest session');
    }

    // Create temporary analysis request record
    const { data: tempRequest, error: requestError } = await supabaseClient
      .from('temp_analysis_requests_2025_11_27_05_00')
      .insert({
        guest_session_id: guestSession.id,
        file_name: fileName,
        file_type: fileType,
        file_size: fileSize,
        file_url: fileUrl,
        original_url: originalUrl,
        status: 'processing'
      })
      .select()
      .single();

    if (requestError) {
      throw requestError;
    }

    // Perform deepfake analysis (same logic as authenticated users)
    const analysisResult = await performDeepfakeAnalysis(fileType, fileUrl || originalUrl);

    // Store temporary analysis results
    const { error: resultError } = await supabaseClient
      .from('temp_analysis_results_2025_11_27_05_00')
      .insert({
        temp_request_id: tempRequest.id,
        manipulation_probability: analysisResult.manipulationProbability,
        confidence_score: analysisResult.confidenceScore,
        detection_method: analysisResult.detectionMethod,
        analysis_data: analysisResult.analysisData,
        anomaly_regions: analysisResult.anomalyRegions,
        frame_analysis: analysisResult.frameAnalysis
      });

    if (resultError) {
      throw resultError;
    }

    // Update temp request status
    await supabaseClient
      .from('temp_analysis_requests_2025_11_27_05_00')
      .update({ status: 'completed' })
      .eq('id', tempRequest.id);

    // Increment guest upload count
    const { data: newCount, error: incrementError } = await supabaseClient
      .rpc('increment_guest_upload_2025_11_27_05_00', { session_id_param: guestSessionId });

    if (incrementError) {
      console.error('Error incrementing upload count:', incrementError);
    }

    return new Response(
      JSON.stringify({
        success: true,
        requestId: tempRequest.id,
        result: analysisResult,
        remainingUploads: Math.max(0, 3 - (newCount || 0)),
        isGuestMode: true,
        expiresAt: tempRequest.expires_at
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error processing guest media file:', error);
    return new Response(
      JSON.stringify({ 
        error: error.message,
        isGuestMode: true
      }),
      { 
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});

async function performDeepfakeAnalysis(fileType: string, fileUrl: string) {
  // Simulate AI analysis with realistic results (same as main function)
  const baseScore = Math.random() * 100;
  
  const result = {
    manipulationProbability: parseFloat(baseScore.toFixed(2)),
    confidenceScore: parseFloat((85 + Math.random() * 10).toFixed(2)),
    detectionMethod: getDetectionMethod(fileType),
    analysisData: generateAnalysisData(fileType),
    anomalyRegions: generateAnomalyRegions(fileType),
    frameAnalysis: fileType === 'video' ? generateFrameAnalysis() : null
  };

  return result;
}

function getDetectionMethod(fileType: string): string {
  const methods = {
    'video': 'FaceSwap Detection + Temporal Consistency Analysis',
    'image': 'GAN Artifact Detection + Pixel-level Analysis',
    'audio': 'Spectral Analysis + Voice Synthesis Detection'
  };
  return methods[fileType] || 'Multi-modal Analysis';
}

function generateAnalysisData(fileType: string) {
  if (fileType === 'video') {
    return {
      totalFrames: Math.floor(Math.random() * 1000) + 100,
      suspiciousFrames: Math.floor(Math.random() * 50) + 5,
      temporalInconsistencies: Math.floor(Math.random() * 20),
      faceSwapIndicators: Math.random() > 0.5
    };
  } else if (fileType === 'image') {
    return {
      pixelAnomalies: Math.floor(Math.random() * 100) + 10,
      compressionArtifacts: Math.random() > 0.3,
      ganSignatures: Math.random() > 0.4,
      metadataInconsistencies: Math.random() > 0.6
    };
  } else if (fileType === 'audio') {
    return {
      spectralAnomalies: Math.floor(Math.random() * 50) + 5,
      voiceSynthesisMarkers: Math.random() > 0.5,
      frequencyInconsistencies: Math.floor(Math.random() * 30),
      prosodyAnalysis: Math.random() > 0.4
    };
  }
  return {};
}

function generateAnomalyRegions(fileType: string) {
  if (fileType === 'image' || fileType === 'video') {
    const regions = [];
    const numRegions = Math.floor(Math.random() * 5) + 1;
    
    for (let i = 0; i < numRegions; i++) {
      regions.push({
        x: Math.floor(Math.random() * 800),
        y: Math.floor(Math.random() * 600),
        width: Math.floor(Math.random() * 200) + 50,
        height: Math.floor(Math.random() * 200) + 50,
        confidence: parseFloat((Math.random() * 100).toFixed(2)),
        type: ['face_swap', 'texture_anomaly', 'lighting_inconsistency'][Math.floor(Math.random() * 3)]
      });
    }
    return regions;
  }
  return [];
}

function generateFrameAnalysis() {
  const frames = [];
  const numFrames = Math.floor(Math.random() * 20) + 10;
  
  for (let i = 0; i < numFrames; i++) {
    frames.push({
      frameNumber: i * 30,
      timestamp: (i * 30 / 30).toFixed(2) + 's',
      manipulationScore: parseFloat((Math.random() * 100).toFixed(2)),
      anomalies: Math.floor(Math.random() * 5)
    });
  }
  return frames;
}