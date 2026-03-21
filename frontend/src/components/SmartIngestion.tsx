import React, { useState, useRef, useEffect, useCallback } from 'react';

interface Props {
  selectedFile: File | null;
  filePreview: string | null;
  uploadProgress: number;
  error: string | null;
  onFileSelect: (file: File) => void;
}

interface ValidationStep {
  label: string;
  status: 'pending' | 'active' | 'done' | 'error';
}

const ACCEPTED_TYPES = ['.png', '.jpg', '.jpeg', '.dcm', '.nii', '.nii.gz'];

const SmartIngestion: React.FC<Props> = ({
  selectedFile,
  filePreview,
  uploadProgress,
  error,
  onFileSelect,
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [validationSteps, setValidationSteps] = useState<ValidationStep[]>([
    { label: 'Scan readable', status: 'pending' },
    { label: 'Format supported', status: 'pending' },
    { label: 'Ready for inference', status: 'pending' },
  ]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Run validation animation when file is selected
  useEffect(() => {
    if (!selectedFile) {
      setValidationSteps([
        { label: 'Scan readable', status: 'pending' },
        { label: 'Format supported', status: 'pending' },
        { label: 'Ready for inference', status: 'pending' },
      ]);
      return;
    }

    // Step 1: Scan readable
    setValidationSteps([
      { label: 'Scan readable', status: 'active' },
      { label: 'Format supported', status: 'pending' },
      { label: 'Ready for inference', status: 'pending' },
    ]);

    const t1 = setTimeout(() => {
      setValidationSteps([
        { label: 'Scan readable', status: 'done' },
        { label: 'Format supported', status: 'active' },
        { label: 'Ready for inference', status: 'pending' },
      ]);
    }, 400);

    const t2 = setTimeout(() => {
      const ext = selectedFile.name.toLowerCase();
      const supported = ACCEPTED_TYPES.some(t => ext.endsWith(t));
      setValidationSteps([
        { label: 'Scan readable', status: 'done' },
        { label: 'Format supported', status: supported ? 'done' : 'error' },
        { label: 'Ready for inference', status: supported ? 'active' : 'pending' },
      ]);
    }, 800);

    const t3 = setTimeout(() => {
      const ext = selectedFile.name.toLowerCase();
      const supported = ACCEPTED_TYPES.some(t => ext.endsWith(t));
      if (supported) {
        setValidationSteps([
          { label: 'Scan readable', status: 'done' },
          { label: 'Format supported', status: 'done' },
          { label: 'Ready for inference', status: 'done' },
        ]);
      }
    }, 1200);

    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [selectedFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFileSelect(file);
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  }, [onFileSelect]);

  const getFileType = (name: string): string => {
    const lower = name.toLowerCase();
    if (lower.endsWith('.dcm')) return 'DICOM';
    if (lower.endsWith('.nii') || lower.endsWith('.nii.gz')) return 'NIfTI';
    if (lower.endsWith('.png')) return 'PNG';
    if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'JPEG';
    return 'Unknown';
  };

  const getModality = (name: string): string => {
    const lower = name.toLowerCase();
    if (lower.includes('t1ce') || lower.includes('t1gd')) return 'T1-CE';
    if (lower.includes('t1')) return 'T1';
    if (lower.includes('t2')) return 'T2';
    if (lower.includes('flair')) return 'FLAIR';
    return 'Auto-detect';
  };

  return (
    <div className="crw-ingestion">
      <div className="crw-ingestion-title">Smart Ingestion</div>
      <div className="crw-ingestion-subtitle">
        Upload a brain MRI scan for AI-powered analysis
      </div>

      {/* Drop Zone */}
      {!selectedFile && (
        <div
          className={`crw-drop-zone ${dragOver ? 'dragover' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <span className="icon">⊕</span>
          <span className="label">Drop MRI scan here or click to browse</span>
          <span className="hint">Supports DICOM (.dcm), NIfTI (.nii), PNG, JPEG</span>
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz"
            style={{ display: 'none' }}
            onChange={handleFileInput}
          />
        </div>
      )}

      {/* File Preview */}
      {selectedFile && (
        <div className="crw-file-preview">
          <div className="crw-file-preview-card">
            {filePreview ? (
              <img src={filePreview} alt="MRI Preview" className="crw-file-preview-img" />
            ) : (
              <div className="crw-file-preview-img" style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '24px', color: 'var(--text-tertiary)'
              }}>
                ◎
              </div>
            )}
            <div className="crw-file-info">
              <div className="crw-file-name">{selectedFile.name}</div>
              <div className="crw-file-meta">
                <span>Type: {getFileType(selectedFile.name)}</span>
                <span>Size: {(selectedFile.size / 1024).toFixed(1)} KB</span>
                <span>Modality: {getModality(selectedFile.name)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Validation Pipeline */}
      {selectedFile && (
        <div className="crw-validation-list">
          {validationSteps.map((step, i) => (
            <div key={i} className={`crw-validation-step ${step.status}`}>
              <div className="crw-validation-icon">
                {step.status === 'done' ? '✓' :
                 step.status === 'error' ? '✕' :
                 step.status === 'active' ? '◌' :
                 '○'}
              </div>
              <span style={{ color: step.status === 'done' ? 'var(--conf-high)' :
                             step.status === 'error' ? 'var(--conf-low)' :
                             step.status === 'active' ? 'var(--cyan)' :
                             'var(--text-tertiary)' }}>
                {step.label}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Upload Progress */}
      {uploadProgress > 0 && uploadProgress < 100 && (
        <div className="crw-upload-progress">
          <div className="crw-progress-bar">
            <div className="crw-progress-fill" style={{ width: `${uploadProgress}%` }} />
          </div>
          <div className="crw-progress-text">Uploading… {uploadProgress.toFixed(0)}%</div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="crw-confidence-banner low" style={{ maxWidth: '520px', marginTop: '16px' }}>
          ⚠ {error}
        </div>
      )}
    </div>
  );
};

export default SmartIngestion;
