import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './Auth';
import { useGuestSession } from '@/hooks/useGuestSession';
import { GuestModeIndicator } from './GuestModeIndicator';
import { 
  Upload, 
  Video, 
  Image, 
  Music, 
  Link, 
  FileText,
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react';

interface UploadedFile {
  file: File;
  type: 'video' | 'image' | 'audio';
  preview?: string;
}

export const FileUpload: React.FC = () => {
  const { user, isGuestMode } = useAuth();
  const { guestSession, updateUploadCount, isLimitReached } = useGuestSession();
  const { toast } = useToast();
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [urlInput, setUrlInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);

  const handleFileUpload = useCallback((files: FileList | null) => {
    if (!files) return;

    const newFiles: UploadedFile[] = [];
    
    Array.from(files).forEach(file => {
      const fileType = getFileType(file.type);
      if (fileType) {
        const uploadedFile: UploadedFile = {
          file,
          type: fileType
        };

        // Create preview for images and videos
        if (fileType === 'image' || fileType === 'video') {
          uploadedFile.preview = URL.createObjectURL(file);
        }

        newFiles.push(uploadedFile);
      } else {
        toast({
          title: "Unsupported File Type",
          description: `${file.name} is not a supported media file`,
          variant: "destructive"
        });
      }
    });

    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, [toast]);

  const getFileType = (mimeType: string): 'video' | 'image' | 'audio' | null => {
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('audio/')) return 'audio';
    return null;
  };

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'video': return <Video className="h-5 w-5" />;
      case 'image': return <Image className="h-5 w-5" />;
      case 'audio': return <Music className="h-5 w-5" />;
      default: return <FileText className="h-5 w-5" />;
    }
  };

  const uploadFileToStorage = async (file: File): Promise<string> => {
    const fileExt = file.name.split('.').pop();
    
    // Choose storage bucket and path based on user type
    const bucketName = isGuestMode ? 'temp-guest-files' : 'media-files';
    const userId = isGuestMode ? (guestSession?.sessionId || 'anonymous') : user?.id;
    const fileName = `${userId}/${Date.now()}.${fileExt}`;
    
    const { data, error } = await supabase.storage
      .from(bucketName)
      .upload(fileName, file);

    if (error) throw error;

    const { data: { publicUrl } } = supabase.storage
      .from(bucketName)
      .getPublicUrl(fileName);

    return publicUrl;
  };

  const processFiles = async () => {
    // Check authentication for registered users or guest mode
    if (!user && !isGuestMode) {
      toast({
        title: "Authentication Required",
        description: "Please sign in or use demo mode to process files",
        variant: "destructive"
      });
      return;
    }

    // Check guest mode limits
    if (isGuestMode && (!guestSession || isLimitReached())) {
      toast({
        title: "Demo Limit Reached",
        description: "You've reached the 3-file limit for demo mode. Please register for unlimited access.",
        variant: "destructive"
      });
      return;
    }

    if (uploadedFiles.length === 0 && !urlInput.trim()) {
      toast({
        title: "No Files Selected",
        description: "Please upload files or enter a URL to analyze",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);

    try {
      const batchId = crypto.randomUUID();
      const totalFiles = uploadedFiles.length + (urlInput.trim() ? 1 : 0);
      let processedFiles = 0;

      // Process uploaded files
      for (const uploadedFile of uploadedFiles) {
        try {
          // Upload file to appropriate storage bucket
          const fileUrl = await uploadFileToStorage(uploadedFile.file);
          
          // Choose the appropriate edge function based on user type
          const functionName = isGuestMode ? 'process_guest_media_file_2025_11_27_05_00' : 'process_media_file_2025_11_27_04_00';
          
          const requestBody: any = {
            fileName: uploadedFile.file.name,
            fileType: uploadedFile.type,
            fileSize: uploadedFile.file.size,
            fileUrl: fileUrl,
            batchId: uploadedFiles.length > 1 ? batchId : undefined
          };

          // Add guest session ID for guest mode
          if (isGuestMode && guestSession) {
            requestBody.guestSessionId = guestSession.sessionId;
          }

          const { data, error } = await supabase.functions.invoke(functionName, {
            body: requestBody
          });

          if (error) throw error;

          // Update guest session upload count if in guest mode
          if (isGuestMode && data.remainingUploads !== undefined) {
            updateUploadCount(3 - data.remainingUploads);
          }

          processedFiles++;
          setProcessingProgress((processedFiles / totalFiles) * 100);

          toast({
            title: "File Processed",
            description: `${uploadedFile.file.name} analysis completed${isGuestMode ? ` (${data.remainingUploads || 0} uploads remaining)` : ''}`,
          });

        } catch (error: any) {
          console.error('Error processing file:', error);
          toast({
            title: "Processing Error",
            description: `Failed to process ${uploadedFile.file.name}: ${error.message}`,
            variant: "destructive"
          });
        }
      }

      // Process URL if provided
      if (urlInput.trim()) {
        try {
          const urlType = getUrlType(urlInput);
          
          const { data, error } = await supabase.functions.invoke('process_media_file_2025_11_27_04_00', {
            body: {
              fileName: `URL_Analysis_${Date.now()}`,
              fileType: urlType,
              fileSize: 0,
              originalUrl: urlInput.trim(),
              batchId: uploadedFiles.length > 0 ? batchId : undefined
            }
          });

          if (error) throw error;

          processedFiles++;
          setProcessingProgress(100);

          toast({
            title: "URL Processed",
            description: "URL analysis completed",
          });

        } catch (error: any) {
          console.error('Error processing URL:', error);
          toast({
            title: "URL Processing Error",
            description: `Failed to process URL: ${error.message}`,
            variant: "destructive"
          });
        }
      }

      // Clear form after successful processing
      setUploadedFiles([]);
      setUrlInput('');
      
      toast({
        title: "Analysis Complete",
        description: "All files have been processed. Check the results in the Analysis Dashboard.",
      });

    } catch (error: any) {
      console.error('Error in batch processing:', error);
      toast({
        title: "Processing Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  };

  const getUrlType = (url: string): 'video' | 'image' | 'audio' => {
    const lowerUrl = url.toLowerCase();
    if (lowerUrl.includes('youtube') || lowerUrl.includes('vimeo') || lowerUrl.includes('.mp4') || lowerUrl.includes('.mov')) {
      return 'video';
    }
    if (lowerUrl.includes('.jpg') || lowerUrl.includes('.png') || lowerUrl.includes('.jpeg')) {
      return 'image';
    }
    return 'audio'; // Default fallback
  };

  const removeFile = (index: number) => {
    setUploadedFiles(prev => {
      const newFiles = [...prev];
      if (newFiles[index].preview) {
        URL.revokeObjectURL(newFiles[index].preview!);
      }
      newFiles.splice(index, 1);
      return newFiles;
    });
  };

  return (
    <div className="space-y-6">
      {/* Guest Mode Indicator */}
      {isGuestMode && <GuestModeIndicator />}
      <Card className="forensic-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Media Upload & Analysis
          </CardTitle>
          <CardDescription>
            Upload video, image, or audio files for deepfake detection analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="upload" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="upload">File Upload</TabsTrigger>
              <TabsTrigger value="url">URL Analysis</TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-4">
              <div className="upload-zone">
                <input
                  type="file"
                  multiple
                  accept="video/*,image/*,audio/*"
                  onChange={(e) => handleFileUpload(e.target.files)}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-lg font-medium mb-2">Drop files here or click to upload</p>
                  <p className="text-sm text-muted-foreground">
                    Supports: MP4, MOV (video) • JPG, PNG (image) • WAV, MP3 (audio)
                  </p>
                </label>
              </div>

              {uploadedFiles.length > 0 && (
                <div className="space-y-2">
                  <h4 className="font-medium">Uploaded Files:</h4>
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-secondary rounded-lg">
                      <div className="flex items-center gap-3">
                        {getFileIcon(file.type)}
                        <div>
                          <p className="font-medium">{file.file.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {file.type.toUpperCase()} • {(file.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => removeFile(index)}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="url" className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="url-input">Media URL</Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Link className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="url-input"
                      placeholder="https://example.com/video.mp4"
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  Enter a direct URL to a video, image, or audio file for analysis
                </p>
              </div>
            </TabsContent>
          </Tabs>

          {isProcessing && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 animate-spin" />
                <span className="text-sm font-medium">Processing files...</span>
              </div>
              <Progress value={processingProgress} className="w-full" />
              <p className="text-xs text-muted-foreground">
                {processingProgress.toFixed(0)}% complete
              </p>
            </div>
          )}

          <Button 
            onClick={processFiles} 
            className="w-full mt-4" 
            disabled={
              isProcessing || 
              (uploadedFiles.length === 0 && !urlInput.trim()) ||
              (isGuestMode && isLimitReached())
            }
          >
            {isProcessing ? 'Processing...' : 
             isGuestMode && isLimitReached() ? 'Demo Limit Reached' :
             'Start Analysis'}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};