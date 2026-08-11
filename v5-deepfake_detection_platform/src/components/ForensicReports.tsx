import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './Auth';
import { 
  FileText, 
  Download, 
  Calendar, 
  User, 
  Building,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';

interface ForensicReport {
  id: string;
  request_id: string;
  report_data: any;
  created_at: string;
  analysis_requests_2025_11_27_04_00: {
    file_name: string;
    file_type: string;
    analysis_results_2025_11_27_04_00: Array<{
      manipulation_probability: number;
    }>;
  };
}

export const ForensicReports: React.FC = () => {
  const { user } = useAuth();
  const { toast } = useToast();
  const [reports, setReports] = useState<ForensicReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedReport, setSelectedReport] = useState<ForensicReport | null>(null);

  useEffect(() => {
    if (user) {
      fetchReports();
    }
  }, [user]);

  const fetchReports = async () => {
    try {
      const { data, error } = await supabase
        .from('forensic_reports_2025_11_27_04_00')
        .select(`
          *,
          analysis_requests_2025_11_27_04_00!inner (
            file_name,
            file_type,
            user_id,
            analysis_results_2025_11_27_04_00 (
              manipulation_probability
            )
          )
        `)
        .eq('analysis_requests_2025_11_27_04_00.user_id', user?.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setReports(data || []);
    } catch (error: any) {
      console.error('Error fetching reports:', error);
      toast({
        title: "Error Loading Reports",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskBadge = (probability: number) => {
    if (probability >= 70) {
      return <Badge className="risk-high"><AlertTriangle className="h-3 w-3 mr-1" />HIGH RISK</Badge>;
    } else if (probability >= 40) {
      return <Badge className="risk-medium"><Clock className="h-3 w-3 mr-1" />MEDIUM RISK</Badge>;
    } else {
      return <Badge className="risk-low"><CheckCircle className="h-3 w-3 mr-1" />LOW RISK</Badge>;
    }
  };

  const downloadReport = (report: ForensicReport) => {
    // In a real implementation, this would download the actual PDF
    const reportContent = JSON.stringify(report.report_data, null, 2);
    const blob = new Blob([reportContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensic_report_${report.report_data.metadata.reportId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Report Downloaded",
      description: "Forensic report has been downloaded successfully",
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Clock className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading forensic reports...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Forensic Reports</h2>
          <p className="text-muted-foreground">Download and review detailed analysis reports</p>
        </div>
        <Button onClick={fetchReports} variant="outline">
          <FileText className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {reports.length === 0 ? (
        <Card className="forensic-card">
          <CardContent className="text-center py-12">
            <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-medium mb-2">No Reports Generated</h3>
            <p className="text-muted-foreground">
              Generate reports from your analysis results to see them here
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {reports.map((report) => {
            const probability = report.analysis_requests_2025_11_27_04_00.analysis_results_2025_11_27_04_00[0]?.manipulation_probability || 0;
            
            return (
              <Card key={report.id} className="forensic-card hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      {report.report_data.metadata.reportId}
                    </CardTitle>
                    {getRiskBadge(probability)}
                  </div>
                  <CardDescription>
                    {report.analysis_requests_2025_11_27_04_00.file_name}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span>Generated: {new Date(report.created_at).toLocaleDateString()}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span>Case: {report.report_data.metadata.caseNumber}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <User className="h-4 w-4 text-muted-foreground" />
                      <span>Analyst: {report.report_data.metadata.generatedBy}</span>
                    </div>
                  </div>

                  <div className="p-3 bg-secondary rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium">Classification:</span>
                      <Badge variant="outline">{report.report_data.metadata.classification}</Badge>
                    </div>
                    <div className="text-sm">
                      <strong>Verdict:</strong> {report.report_data.detectionResults.verdict}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={() => setSelectedReport(report)}
                      className="flex-1"
                    >
                      View Details
                    </Button>
                    <Button 
                      size="sm" 
                      onClick={() => downloadReport(report)}
                      className="flex-1"
                    >
                      <Download className="h-4 w-4 mr-1" />
                      Download
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Detailed Report View */}
      {selectedReport && (
        <Card className="forensic-card mt-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Forensic Report: {selectedReport.report_data.metadata.reportId}
              </CardTitle>
              <Button variant="outline" onClick={() => setSelectedReport(null)}>
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Report Metadata */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-3 flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Case Information
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Report ID:</span>
                      <span className="font-medium">{selectedReport.report_data.metadata.reportId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Case Number:</span>
                      <span className="font-medium">{selectedReport.report_data.metadata.caseNumber}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Classification:</span>
                      <Badge variant="outline">{selectedReport.report_data.metadata.classification}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Generated:</span>
                      <span className="font-medium">
                        {new Date(selectedReport.report_data.metadata.generatedAt).toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-3 flex items-center gap-2">
                    <Building className="h-4 w-4" />
                    File Analysis
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>File Name:</span>
                      <span className="font-medium">{selectedReport.report_data.fileAnalysis.fileName}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>File Type:</span>
                      <span className="font-medium">{selectedReport.report_data.fileAnalysis.fileType.toUpperCase()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>File Size:</span>
                      <span className="font-medium">{(selectedReport.report_data.fileAnalysis.fileSize / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Processing Time:</span>
                      <span className="font-medium">{selectedReport.report_data.fileAnalysis.processingTime}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-3 flex items-center gap-2">
                    <Shield className="h-4 w-4" />
                    Detection Results
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Manipulation Probability:</span>
                      <span className="font-medium">{selectedReport.report_data.detectionResults.manipulationProbability}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Confidence Score:</span>
                      <span className="font-medium">{selectedReport.report_data.detectionResults.confidenceScore}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Detection Method:</span>
                      <span className="font-medium text-xs">{selectedReport.report_data.detectionResults.detectionMethod}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Verdict:</span>
                      <Badge variant={selectedReport.report_data.detectionResults.verdict.includes('MANIPULATED') ? 'destructive' : 'default'}>
                        {selectedReport.report_data.detectionResults.verdict}
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium mb-3">Chain of Custody</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Uploaded:</span>
                      <span className="font-medium">
                        {new Date(selectedReport.report_data.chainOfCustody.uploadedAt).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Processed:</span>
                      <span className="font-medium">
                        {new Date(selectedReport.report_data.chainOfCustody.processedAt).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Integrity:</span>
                      <Badge className="risk-low">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        {selectedReport.report_data.chainOfCustody.integrity}
                      </Badge>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="p-4 bg-secondary rounded-lg">
              <h4 className="font-medium mb-3">Forensic Recommendations</h4>
              <ul className="space-y-2 text-sm">
                {selectedReport.report_data.recommendations.map((recommendation: string, index: number) => (
                  <li key={index} className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2 flex-shrink-0" />
                    <span>{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="flex justify-end">
              <Button onClick={() => downloadReport(selectedReport)}>
                <Download className="h-4 w-4 mr-2" />
                Download Full Report
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};