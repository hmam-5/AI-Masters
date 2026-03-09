// React component for brain MRI image upload.

import React, { useState, useRef } from "react";
import apiService from "../services/api";
import type { ImageAnalyzeResponse } from "../types/api";

interface UploadFormProps {
  onAnalysisStarted?: (response: ImageAnalyzeResponse) => void;
  onError?: (error: string) => void;
}

export const UploadForm: React.FC<UploadFormProps> = ({
  onAnalysisStarted,
  onError,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    if (selectedFile.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) handleFile(selected);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFile(dropped);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image file");
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const response = await apiService.analyzeImage(file, (progress) => {
        setUploadProgress(progress);
      });
      onAnalysisStarted?.(response);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Upload failed";
      setError(errorMsg);
      onError?.(errorMsg);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div style={{ padding: "36px 32px" }}>
      <style>{`
        @keyframes uploadPulse { 0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,0.25)} 50%{box-shadow:0 0 0 12px rgba(99,102,241,0)} }
        @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
        .drop-zone:hover { border-color: #818cf8 !important; background: rgba(99,102,241,0.04) !important; }
      `}</style>

      <form onSubmit={handleSubmit}>
        {/* Drop Zone */}
        <div
          className="drop-zone"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${isDragOver ? "#6366f1" : file ? "#10b981" : "#cbd5e1"}`,
            borderRadius: "16px",
            padding: preview ? "20px" : "48px 24px",
            textAlign: "center",
            cursor: "pointer",
            backgroundColor: isDragOver ? "rgba(99,102,241,0.06)" : file ? "rgba(16,185,129,0.04)" : "rgba(248,250,252,0.6)",
            transition: "all 0.4s cubic-bezier(0.4,0,0.2,1)",
            marginBottom: "24px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg"
            onChange={handleFileChange}
            disabled={isUploading}
            style={{ display: "none" }}
          />

          {preview ? (
            <div style={{ animation: "fadeIn 0.5s ease" }}>
              <div style={{
                position: "relative",
                display: "inline-block",
                borderRadius: "12px",
                overflow: "hidden",
                boxShadow: "0 8px 32px rgba(0,0,0,0.12)",
              }}>
                <img
                  src={preview}
                  alt="Preview"
                  style={{
                    maxWidth: "100%",
                    maxHeight: "260px",
                    display: "block",
                  }}
                />
                <div style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  background: "linear-gradient(transparent, rgba(0,0,0,0.7))",
                  padding: "20px 16px 12px",
                  color: "#fff",
                }}>
                  <p style={{ margin: 0, fontSize: "13px", fontWeight: 600 }}>
                    {file?.name}
                  </p>
                  <p style={{ margin: "2px 0 0", fontSize: "11px", opacity: 0.8 }}>
                    {((file?.size || 0) / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <div style={{ marginTop: "12px" }}>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); clearFile(); }}
                  style={{
                    padding: "6px 16px",
                    background: "transparent",
                    color: "#ef4444",
                    border: "1px solid #fca5a5",
                    borderRadius: "8px",
                    cursor: "pointer",
                    fontSize: "12px",
                    fontWeight: 500,
                    transition: "all 0.2s",
                  }}
                >
                  Remove Image
                </button>
              </div>
            </div>
          ) : (
            <div>
              <div style={{
                width: "72px",
                height: "72px",
                margin: "0 auto 16px",
                borderRadius: "20px",
                background: "linear-gradient(135deg, #ede9fe 0%, #e0e7ff 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "32px",
                transition: "transform 0.3s",
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <p style={{ margin: "0 0 6px", fontSize: "16px", color: "#1e293b", fontWeight: 600 }}>
                Drop your MRI image here
              </p>
              <p style={{ margin: "0 0 12px", color: "#94a3b8", fontSize: "13px" }}>
                or click to browse your files
              </p>
              <div style={{
                display: "inline-flex",
                gap: "6px",
                padding: "6px 14px",
                borderRadius: "20px",
                backgroundColor: "rgba(99,102,241,0.08)",
              }}>
                {["PNG", "JPG", "JPEG"].map(fmt => (
                  <span key={fmt} style={{
                    fontSize: "11px",
                    color: "#6366f1",
                    fontWeight: 600,
                    letterSpacing: "0.5px",
                  }}>{fmt}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Upload Progress */}
        {isUploading && (
          <div style={{ marginBottom: "20px" }}>
            <div style={{
              width: "100%",
              height: "6px",
              backgroundColor: "rgba(99,102,241,0.12)",
              borderRadius: "3px",
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%",
                width: `${uploadProgress}%`,
                background: "linear-gradient(90deg, #6366f1, #8b5cf6, #6366f1)",
                backgroundSize: "200% 100%",
                animation: "shimmer 1.5s linear infinite",
                transition: "width 0.3s",
                borderRadius: "3px",
              }} />
            </div>
            <p style={{ marginTop: "8px", fontSize: "12px", color: "#64748b", textAlign: "center", fontWeight: 500 }}>
              Uploading... {uploadProgress.toFixed(0)}%
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            marginBottom: "20px",
            padding: "12px 16px",
            background: "linear-gradient(135deg, #fef2f2, #fff1f2)",
            color: "#dc2626",
            borderRadius: "10px",
            border: "1px solid #fecaca",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}>
            <span style={{ fontSize: "16px" }}>&#9888;</span> {error}
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={isUploading || !file}
          style={{
            width: "100%",
            padding: "16px 24px",
            background: isUploading || !file
              ? "linear-gradient(135deg, #cbd5e1, #94a3b8)"
              : "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
            color: "#fff",
            border: "none",
            borderRadius: "12px",
            cursor: isUploading || !file ? "not-allowed" : "pointer",
            fontSize: "15px",
            fontWeight: 600,
            letterSpacing: "0.3px",
            transition: "all 0.3s cubic-bezier(0.4,0,0.2,1)",
            boxShadow: isUploading || !file ? "none" : "0 4px 14px rgba(99,102,241,0.4)",
            animation: file && !isUploading ? "uploadPulse 2s infinite" : "none",
          }}
        >
          {isUploading ? (
            <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ animation: "spin 1s linear infinite" }}>
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
              </svg>
              Analyzing...
            </span>
          ) : (
            <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
              Start Analysis
            </span>
          )}
        </button>
      </form>
    </div>
  );
};

export default UploadForm;
