// TypeScript types for medical imaging API.

export interface Patient {
  id: string;
  mrn: string;
  dateOfBirth: string;
  sex?: string;
  createdAt: string;
  updatedAt: string;
}

export interface MRIScan {
  id: string;
  patientId: string;
  scanDate: string;
  modalities: string[];
  status: string;
  preprocessingComplete: boolean;
  imageShape?: [number, number, number];
  createdAt: string;
  updatedAt: string;
}

export interface FileUpload {
  filename: string;
  sizeBytes: number;
  storagePath: string;
  uploadTimestamp: string;
}

export interface MRIScanUpload {
  scanId: string;
  patientId: string;
  uploadedModalities: Record<string, FileUpload>;
  totalSizeMb: number;
  timestamp: string;
}

export type JobStatus = "pending" | "processing" | "completed" | "failed" | "cancelled";

export interface InferenceJob {
  id: string;
  scanId: string;
  celeryTaskId?: string;
  status: JobStatus;
  progressPercentage: number;
  startedAt?: string;
  completedAt?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
}

export interface SegmentationResult {
  id: string;
  jobId: string;
  subregion: "enhancing_tumor" | "edema" | "necrotic_core";
  confidenceScore: number;
  volumeMm3?: number;
  maskStoragePath: string;
  diceCoefficient?: number;
  hausdorffDistance?: number;
  createdAt: string;
}

export interface ClassificationResult {
  id: string;
  jobId: string;
  tumorGrade: string;
  confidenceScore: number;
  classificationDetails?: Record<string, unknown>;
  createdAt: string;
}

export interface WebSocketMessage {
  event: string;
  jobId: string;
  status?: JobStatus;
  progress?: number;
  timestamp: string;
}

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
