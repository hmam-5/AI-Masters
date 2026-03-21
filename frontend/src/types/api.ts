// TypeScript types for medical imaging API.

export type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";

export interface ImageAnalyzeResponse {
  job_id: string;
  scan_id: string;
  filename: string;
  status: string;
  message: string;
}

export interface AnalysisResult {
  job_id: string;
  status: string;
  image_filename: string;
  tumor_detected: boolean;
  tumor_type?: string;
  tumor_grade?: string;
  confidence: number;
  findings: string[];
  explanation: string;
  recommendations: string[];
  classification_details?: Record<string, unknown>;
  segmentation_summary?: Record<string, { confidence: number; volume_mm3: number }>;
  completed_at?: string;
}
