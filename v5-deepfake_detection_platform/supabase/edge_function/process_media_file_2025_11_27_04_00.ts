import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Authorization, X-Client-Info, apikey, Content-Type, X-Application-Name',
};

interface AnalysisRequest {
  fileName: string;
  fileType: string;
  fileSize: number;
  fileUrl?: string;
  originalUrl?: string;
  batchId?: string;
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

    const authHeader = req.headers.get('Authorization')!;
    const token = authHeader.replace('Bearer ', '');
    const { data: { user } } = await supabaseClient.auth.getUser(token);

    if (!user) {
      throw new Error('Unauthorized');
    }

    const { fileName, fileType, fileSize, fileUrl, originalUrl, batchId }: AnalysisRequest = await req.json();

    // Create analysis request record
    const { data: request, error: requestError } = await supabaseClient
      .from('analysis_requests_2025_11_27_04_00')
      .insert({
        user_id: user.id,
        file_name: fileName,
        file_type: fileType,
        file_size: fileSize,
        file_url: fileUrl,
        original_url: originalUrl,
        batch_id: batchId,
        status: 'processing'
      })
      .select()
      .single();

    if (requestError) {
      throw requestError;
    }

    // Simulate deepfake analysis (in production, this would call actual AI models)
    const analysisResult = await performDeepfakeAnalysis(fileType, fileUrl || originalUrl);

    // Store analysis results
    const { error: resultError } = await supabaseClient
      .from('analysis_results_2025_11_27_04_00')
      .insert({
        request_id: request.id,
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

    // Update request status
    await supabaseClient
      .from('analysis_requests_2025_11_27_04_00')
      .update({ status: 'completed' })
      .eq('id', request.id);

    return new Response(
      JSON.stringify({
        success: true,
        requestId: request.id,
        result: analysisResult
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error processing media file:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});

async function performDeepfakeAnalysis(fileType: string, fileUrl: string) {
  // Simulate AI analysis with realistic results
  const baseScore = Math.random() * 100;
  const isLikelyFake = baseScore > 60;
  
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
      frameNumber: i * 30, // Every 30th frame
      timestamp: (i * 30 / 30).toFixed(2) + 's',
      manipulationScore: parseFloat((Math.random() * 100).toFixed(2)),
      anomalies: Math.floor(Math.random() * 5)
    });
  }
  return frames;
}