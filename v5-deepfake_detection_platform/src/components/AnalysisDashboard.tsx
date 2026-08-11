import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './Auth';
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Download,
  Eye,
  FileText,
  BarChart3,
  Zap,
  Shield,
  Activity
} from 'lucide-react';

interface AnalysisRequest {
  id: string;
  file_name: string;
  file_type: string;
  file_size: number;
  status: string;
  created_at: string;
  analysis_results_2025_11_27_04_00: AnalysisResult[];
}

interface AnalysisResult {
  id: string;
  manipulation_probability: number;
  confidence_score: number;
  detection_method: string;
  analysis_data: any;
  anomaly_regions: any[];
  frame_analysis: any[];
  created_at: string;
}

export const AnalysisDashboard: React.FC = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [requests, setRequests] = useState<AnalysisRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRequest, setSelectedRequest] = useState<AnalysisRequest | null>(null);

  useEffect(() => {
    if (user) {
      fetchAnalysisRequests();
    }
  }, [user]);

  const fetchAnalysisRequests = async () => {
    try {
      const { data, error } = await supabase
        .from('analysis_requests_2025_11_27_04_00')
        .select(`
          *,
          analysis_results_2025_11_27_04_00 (*)
        `)
        .eq('user_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setRequests(data || []);
    } catch (error: any) {
      console.error('Error fetching requests:', error);
      toast({
        title: "Error Loading Data",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (probability: number) => {
    if (probability >= 70) return { level: 'HIGH', color: 'risk-high', icon: AlertTriangle };
    if (probability >= 40) return { level: 'MEDIUM', color: 'risk-medium', icon: Activity };
    return { level: 'LOW', color: 'risk-low', icon: CheckCircle };
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing': return <Clock className="h-4 w-4 text-yellow-500 animate-spin" />;
      case 'failed': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const generateReport = async (requestId: string) => {
    try {
      const { data, error } = await supabase.functions.invoke('generate_forensic_report_2025_11_27_04_00', {
        body: { requestId }
      });

      if (error) throw error;

      toast({
        title: "Report Generated",
        description: "Forensic report has been generated successfully",
      });

      // In a real implementation, this would trigger a download
      console.log('Report data:', data);
      
    } catch (error: any) {
      console.error('Error generating report:', error);
      toast({
        title: "Report Generation Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Clock className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading analysis results...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Analysis Dashboard</h2>
          <p className="text-muted-foreground">Review deepfake detection results and generate reports</p>
        </div>
        <Button onClick={fetchAnalysisRequests} variant="outline">
          <Activity className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {requests.length === 0 ? (
        <Card className="forensic-card">
          <CardContent className="text-center py-12">
            <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-medium mb-2">No Analysis Results</h3>
            <p className="text-muted-foreground">
              Upload and analyze media files to see results here
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="analysis-grid">
          {requests.map((request) => {
            const result = request.analysis_results_2025_11_27_04_00[0];
            const risk = result ? getRiskLevel(result.manipulation_probability) : null;
            const RiskIcon = risk?.icon || Clock;

            return (
              <Card key={request.id} className="forensic-card hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg truncate">{request.file_name}</CardTitle>
                    {getStatusIcon(request.status)}
                  </div>
                  <CardDescription>
                    {request.file_type.toUpperCase()} • {formatFileSize(request.file_size)}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {result ? (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Manipulation Risk:</span>
                        <Badge className={`risk-indicator ${risk?.color}`}>
                          <RiskIcon className="h-3 w-3 mr-1" />
                          {risk?.level}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Probability:</span>
                          <span className="font-medium">{result.manipulation_probability}%</span>
                        </div>
                        <Progress value={result.manipulation_probability} className="h-2" />
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span>Confidence:</span>
                          <span className="font-medium">{result.confidence_score}%</span>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Method: {result.detection_method}
                        </div>
                      </div>

                      <div className="flex gap-2 pt-2">
                        <Button 
                          size="sm" 
                          variant="outline" 
                          onClick={() => setSelectedRequest(request)}
                          className="flex-1"
                        >
                          <Eye className="h-4 w-4 mr-1" />
                          Details
                        </Button>
                        <Button 
                          size="sm" 
                          onClick={() => generateReport(request.id)}
                          className="flex-1"
                        >
                          <Download className="h-4 w-4 mr-1" />
                          Report
                        </Button>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-4">
                      <Clock className="h-6 w-6 mx-auto mb-2 text-muted-foreground animate-spin" />
                      <p className="text-sm text-muted-foreground">
                        {request.status === 'processing' ? 'Analyzing...' : 'Pending Analysis'}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Detailed Analysis Modal/Panel */}
      {selectedRequest && selectedRequest.analysis_results_2025_11_27_04_00[0] && (
        <Card className="forensic-card mt-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Detailed Analysis: {selectedRequest.file_name}</CardTitle>
              <Button variant="outline" onClick={() => setSelectedRequest(null)}>
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="technical">Technical</TabsTrigger>
                <TabsTrigger value="regions">Anomalies</TabsTrigger>
                <TabsTrigger value="frames">Frames</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4">
                {(() => {
                  const result = selectedRequest.analysis_results_2025_11_27_04_00[0];
                  const risk = getRiskLevel(result.manipulation_probability);
                  const RiskIcon = risk.icon;
                  
                  return (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="p-4 bg-secondary rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Shield className="h-5 w-5" />
                            <h4 className="font-medium">Detection Summary</h4>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span>Risk Level:</span>
                              <Badge className={`risk-indicator ${risk.color}`}>
                                <RiskIcon className="h-3 w-3 mr-1" />
                                {risk.level}
                              </Badge>
                            </div>
                            <div className="flex justify-between">
                              <span>Manipulation Probability:</span>
                              <span className="font-medium">{result.manipulation_probability}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Confidence Score:</span>
                              <span className="font-medium">{result.confidence_score}%</span>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-secondary rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Zap className="h-5 w-5" />
                            <h4 className="font-medium">Detection Method</h4>
                          </div>
                          <p className="text-sm">{result.detection_method}</p>
                        </div>
                      </div>

                      <div className="space-y-4">
                        <div className="p-4 bg-secondary rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <BarChart3 className="h-5 w-5" />
                            <h4 className="font-medium">Analysis Metrics</h4>
                          </div>
                          <div className="space-y-2 text-sm">
                            {result.analysis_data && Object.entries(result.analysis_data).map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                                <span className="font-medium">{String(value)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </TabsContent>

              <TabsContent value="technical" className="space-y-4">
                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-2">Technical Analysis Data</h4>
                  <pre className="text-xs bg-background p-3 rounded border overflow-auto max-h-64">
                    {JSON.stringify(selectedRequest.analysis_results_2025_11_27_04_00[0].analysis_data, null, 2)}
                  </pre>
                </div>
              </TabsContent>

              <TabsContent value="regions" className="space-y-4">
                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-2">Anomaly Regions</h4>
                  {selectedRequest.analysis_results_2025_11_27_04_00[0].anomaly_regions?.length > 0 ? (
                    <div className="space-y-2">
                      {selectedRequest.analysis_results_2025_11_27_04_00[0].anomaly_regions.map((region: any, index: number) => (
                        <div key={index} className="p-3 bg-background rounded border">
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-medium text-sm">Region {index + 1}</span>
                            <Badge variant="outline">{region.type}</Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            Position: ({region.x}, {region.y}) • Size: {region.width}×{region.height} • Confidence: {region.confidence}%
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No anomaly regions detected</p>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="frames" className="space-y-4">
                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-2">Frame Analysis</h4>
                  {selectedRequest.analysis_results_2025_11_27_04_00[0].frame_analysis?.length > 0 ? (
                    <div className="space-y-2 max-h-64 overflow-auto">
                      {selectedRequest.analysis_results_2025_11_27_04_00[0].frame_analysis.map((frame: any, index: number) => (
                        <div key={index} className="p-3 bg-background rounded border">
                          <div className="flex justify-between items-center">
                            <span className="font-medium text-sm">Frame {frame.frameNumber}</span>
                            <span className="text-xs text-muted-foreground">{frame.timestamp}</span>
                          </div>
                          <div className="flex justify-between text-xs mt-1">
                            <span>Manipulation Score: {frame.manipulationScore}%</span>
                            <span>Anomalies: {frame.anomalies}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      {selectedRequest.file_type === 'video' ? 'No frame analysis available' : 'Frame analysis not applicable for this file type'}
                    </p>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
};