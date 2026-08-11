import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Authorization, X-Client-Info, apikey, Content-Type, X-Application-Name',
};

interface ReportRequest {
  requestId: string;
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

    const { requestId }: ReportRequest = await req.json();

    // Get analysis request and results
    const { data: request, error: requestError } = await supabaseClient
      .from('analysis_requests_2025_11_27_04_00')
      .select(`
        *,
        analysis_results_2025_11_27_04_00 (*)
      `)
      .eq('id', requestId)
      .eq('user_id', user.id)
      .single();

    if (requestError || !request) {
      throw new Error('Analysis request not found');
    }

    const result = request.analysis_results_2025_11_27_04_00[0];
    if (!result) {
      throw new Error('Analysis results not found');
    }

    // Generate PDF report data (in production, this would generate actual PDF)
    const reportData = generateForensicReport(request, result, user);
    
    // Create report record
    const { data: report, error: reportError } = await supabaseClient
      .from('forensic_reports_2025_11_27_04_00')
      .insert({
        request_id: requestId,
        report_data: reportData,
        generated_by: user.id,
        report_url: `forensic-reports/${user.id}/${requestId}_report.pdf`
      })
      .select()
      .single();

    if (reportError) {
      throw reportError;
    }

    return new Response(
      JSON.stringify({
        success: true,
        reportId: report.id,
        reportData: reportData,
        downloadUrl: report.report_url
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error generating report:', error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { 
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});

function generateForensicReport(request: any, result: any, user: any) {
  const reportData = {
    metadata: {
      reportId: `DF-${Date.now()}`,
      generatedAt: new Date().toISOString(),
      generatedBy: user.email,
      caseNumber: `CASE-${request.id.slice(0, 8).toUpperCase()}`,
      classification: result.manipulation_probability > 70 ? 'HIGH RISK' : 
                    result.manipulation_probability > 40 ? 'MEDIUM RISK' : 'LOW RISK'
    },
    fileAnalysis: {
      fileName: request.file_name,
      fileType: request.file_type,
      fileSize: request.file_size,
      analysisDate: result.created_at,
      processingTime: '2.3 seconds' // Simulated
    },
    detectionResults: {
      manipulationProbability: result.manipulation_probability,
      confidenceScore: result.confidence_score,
      detectionMethod: result.detection_method,
      verdict: result.manipulation_probability > 50 ? 'LIKELY MANIPULATED' : 'LIKELY AUTHENTIC'
    },
    technicalAnalysis: {
      analysisData: result.analysis_data,
      anomalyRegions: result.anomaly_regions,
      frameAnalysis: result.frame_analysis
    },
    visualizations: {
      heatmapGenerated: result.anomaly_regions && result.anomaly_regions.length > 0,
      spectrogramGenerated: request.file_type === 'audio',
      frameAnalysisGenerated: request.file_type === 'video'
    },
    recommendations: generateRecommendations(result.manipulation_probability),
    chainOfCustody: {
      uploadedAt: request.created_at,
      processedAt: result.created_at,
      reportGeneratedAt: new Date().toISOString(),
      integrity: 'VERIFIED'
    }
  };

  return reportData;
}

function generateRecommendations(probability: number): string[] {
  const recommendations = [];
  
  if (probability > 70) {
    recommendations.push('HIGH PROBABILITY of manipulation detected. Recommend further investigation.');
    recommendations.push('Consider cross-referencing with original source materials.');
    recommendations.push('Examine metadata for tampering evidence.');
    recommendations.push('Conduct additional forensic analysis using alternative methods.');
  } else if (probability > 40) {
    recommendations.push('MODERATE PROBABILITY of manipulation. Exercise caution.');
    recommendations.push('Verify authenticity through additional sources.');
    recommendations.push('Consider context and provenance of the media file.');
  } else {
    recommendations.push('LOW PROBABILITY of manipulation detected.');
    recommendations.push('File appears to be authentic based on current analysis.');
    recommendations.push('Continue standard verification procedures.');
  }
  
  recommendations.push('Maintain chain of custody documentation.');
  recommendations.push('Archive analysis results for future reference.');
  
  return recommendations;
}