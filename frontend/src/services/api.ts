// API service for backend communication.

import type {
  AnalysisResult,
  ClassificationResult,
  ImageAnalyzeResponse,
  InferenceJob,
  MRIScan,
  MRIScanUpload,
  Patient,
  SegmentationResult,
} from "../types/api";

const API_BASE_URL = process.env.REACT_APP_API_URL || "/api/v1";

class ApiService {
  private async request<T>(
    method: "GET" | "POST" | "PUT" | "DELETE",
    endpoint: string,
    data?: FormData | Record<string, unknown>,
  ): Promise<T> {
    const headers: Record<string, string> = {};

    // Don't set Content-Type for FormData (let browser handle it)
    if (!(data instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method,
        headers,
        body: data ? (data instanceof FormData ? data : JSON.stringify(data)) : undefined,
      });

      if (response.status === 202) {
        const body = await response.json();
        throw new Error(body.detail || "still in progress");
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || error.message || "API Error");
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${method} ${endpoint}]:`, error);
      throw error;
    }
  }

  // Patient endpoints
  async createPatient(mrn: string, dateOfBirth: string, sex?: string): Promise<Patient> {
    return this.request<Patient>("POST", "/patients", {
      mrn,
      date_of_birth: dateOfBirth,
      sex,
    });
  }

  async getPatient(mrn: string): Promise<Patient> {
    return this.request<Patient>("GET", `/patients/${mrn}`);
  }

  // MRI Scan endpoints
  async uploadMRIScan(
    patientId: string,
    files: File[],
    onProgress?: (progress: number) => void,
  ): Promise<MRIScanUpload> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          onProgress?.(progress);
        }
      });

      xhr.addEventListener("load", async () => {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Upload failed"));
      });

      xhr.open("POST", `${API_BASE_URL}/scans/upload?patient_id=${patientId}`);
      xhr.send(formData);
    });
  }

  // Inference endpoints
  async startInference(scanId: string): Promise<InferenceJob> {
    return this.request<InferenceJob>("POST", "/inference/start", {
      scan_id: scanId,
    });
  }

  async getJobStatus(jobId: string): Promise<InferenceJob> {
    return this.request<InferenceJob>("GET", `/inference/${jobId}`);
  }

  // WebSocket connection
  subscribeToJobStatus(
    jobId: string,
    onMessage: (message: any) => void,
    onError?: (error: Event) => void,
  ): WebSocket {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/job/${jobId}`;

    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      onError?.(error);
    };

    return ws;
  }

  // Health check
  async healthCheck(): Promise<{ status: string }> {
    return this.request("GET", "/health");
  }

  // Image Analysis (new simplified flow)
  async analyzeImage(
    file: File,
    onProgress?: (progress: number) => void,
  ): Promise<ImageAnalyzeResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          onProgress?.(progress);
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(err.detail || "Upload failed"));
          } catch {
            reject(new Error(`Upload failed: ${xhr.statusText}`));
          }
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Upload failed - check your connection"));
      });

      xhr.open("POST", `${API_BASE_URL}/analyze`);
      xhr.send(formData);
    });
  }

  async getAnalysisResults(jobId: string): Promise<AnalysisResult> {
    return this.request<AnalysisResult>("GET", `/inference/${jobId}/results`);
  }
}

export default new ApiService();
