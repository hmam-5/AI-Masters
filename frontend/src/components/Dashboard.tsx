import React, { useState, useCallback, useRef, useEffect } from 'react';
import '../styles/crw.css';
import apiService from '../services/api';
import type { AnalysisResult } from '../types/api';
import CommandEntry from './CommandEntry';
import SmartIngestion from './SmartIngestion';
import PipelineView from './PipelineView';
import DiagnosticWorkspace from './DiagnosticWorkspace';
import AIChatAssistant from './AIChatAssistant';
import PipelineLog from './PipelineLog';

/* ── Types ─────────────────────────────────────────────────── */
export type AppScreen = 'command' | 'ingestion' | 'pipeline' | 'workspace';
export type Theme = 'dark' | 'light';

export interface PipelineStage {
  id: string;
  label: string;
  description: string;
  status: 'pending' | 'active' | 'done';
  icon: string;
}

export interface ModelPrediction {
  name: string;
  displayName: string;
  predictedClass: string;
  confidence: number;
  status: 'idle' | 'running' | 'done';
}

export interface LogEntry {
  time: string;
  level: 'info' | 'success' | 'warn' | 'error';
  message: string;
}

export interface SystemStatus {
  modelsOperational: number;
  totalModels: number;
  lastTraining: string;
  lastInference: string;
  systemHealth: 'online' | 'degraded' | 'offline';
  casesAnalyzed: number;
}

/* ── Sidebar Items ─────────────────────────────────────────── */
const SIDEBAR_NAV = [
  { id: 'command', icon: '⊞', label: 'Dashboard' },
  { id: 'ingestion', icon: '⬆', label: 'New Scan' },
  { id: 'workspace', icon: '◎', label: 'Workspace' },
] as const;

const SIDEBAR_TOOLS = [
  { id: 'history', icon: '◷', label: 'Case History' },
  { id: 'similar', icon: '⊶', label: 'Similar Cases' },
  { id: 'monitor', icon: '⊟', label: 'Model Monitor' },
  { id: 'settings', icon: '⚙', label: 'Settings' },
] as const;

/* ── Initial pipeline stages ───────────────────────────────── */
const INITIAL_STAGES: PipelineStage[] = [
  { id: 'queued', label: 'Job Queued', description: 'Analysis request submitted to processing queue', status: 'pending', icon: '◌' },
  { id: 'preprocessing', label: 'Preprocessing', description: 'Image normalization, resizing, and slice preparation', status: 'pending', icon: '⊡' },
  { id: 'tta', label: 'Test-Time Augmentation', description: 'Generating augmented variants (flip, rotate, crop)', status: 'pending', icon: '⊞' },
  { id: 'inference', label: 'Model Inference', description: 'Running 4 parallel neural network predictions', status: 'pending', icon: '◈' },
  { id: 'fusion', label: 'Ensemble Fusion', description: 'Weighted combination of model predictions', status: 'pending', icon: '⊕' },
  { id: 'gate', label: 'Confidence Gate', description: 'Applying confidence thresholds and safety checks', status: 'pending', icon: '⊘' },
];

const INITIAL_MODELS: ModelPrediction[] = [
  { name: 'custom_cnn', displayName: 'Custom CNN', predictedClass: '', confidence: 0, status: 'idle' },
  { name: 'resnet50', displayName: 'ResNet-50', predictedClass: '', confidence: 0, status: 'idle' },
  { name: 'efficientnet', displayName: 'EfficientNet-B0', predictedClass: '', confidence: 0, status: 'idle' },
  { name: 'densenet', displayName: 'DenseNet-121', predictedClass: '', confidence: 0, status: 'idle' },
];

/* ── Dashboard Component ───────────────────────────────────── */
const Dashboard: React.FC = () => {
  // Core state
  const [screen, setScreen] = useState<AppScreen>('command');
  const [theme, setTheme] = useState<Theme>(() => {
    return (document.documentElement.getAttribute('data-theme') as Theme) || 'dark';
  });
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [bottomOpen, setBottomOpen] = useState(true);

  // Analysis state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Pipeline state
  const [stages, setStages] = useState<PipelineStage[]>(INITIAL_STAGES);
  const [models, setModels] = useState<ModelPrediction[]>(INITIAL_MODELS);
  const [agreementScore, setAgreementScore] = useState(0);

  // System status
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    modelsOperational: 4,
    totalModels: 4,
    lastTraining: 'Ready',
    lastInference: 'None',
    systemHealth: 'online',
    casesAnalyzed: 0,
  });

  // Logs
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Add log entry
  const addLog = useCallback((level: LogEntry['level'], message: string) => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setLogs(prev => [...prev.slice(-200), { time, level, message }]);
  }, []);

  // Theme management
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(t => t === 'dark' ? 'light' : 'dark');
  }, []);

  // System health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiService.healthCheck();
        setSystemStatus(s => ({ ...s, systemHealth: 'online' }));
        addLog('info', 'System health check passed — all services operational');
      } catch {
        setSystemStatus(s => ({ ...s, systemHealth: 'degraded' }));
        addLog('warn', 'System health check failed — some services may be unavailable');
      }
    };

    // Fetch training history
    const fetchTraining = async () => {
      try {
        const resp = await fetch('/api/v1/analytics/training');
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.length > 0) {
            const last = data[data.length - 1];
            setSystemStatus(s => ({
              ...s,
              lastTraining: last.completed_at ? new Date(last.completed_at).toLocaleDateString() : 'Ready',
              casesAnalyzed: data.length,
            }));
          }
        }
      } catch { /* ignore */ }
    };

    checkHealth();
    fetchTraining();
  }, [addLog]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  /* ── Pipeline simulation driven by polling ───────────────── */
  const advanceStage = useCallback((stageIndex: number) => {
    setStages(prev => prev.map((s, i) => {
      if (i < stageIndex) return { ...s, status: 'done' };
      if (i === stageIndex) return { ...s, status: 'active' };
      return { ...s, status: 'pending' };
    }));
  }, []);

  const startPipelineTracking = useCallback((jid: string) => {
    let currentStage = 0;
    const stageTimings = [800, 2000, 2500, 3000, 2000, 1500];
    let elapsed = 0;

    addLog('info', `Job ${jid.substring(0, 8)}… queued for processing`);
    advanceStage(0);

    // Model activation during inference stage
    const activateModels = () => {
      const modelNames = ['custom_cnn', 'resnet50', 'efficientnet', 'densenet'];
      modelNames.forEach((name, i) => {
        setTimeout(() => {
          setModels(prev => prev.map(m =>
            m.name === name ? { ...m, status: 'running' } : m
          ));
          addLog('info', `${INITIAL_MODELS[i].displayName} — inference started`);
        }, i * 400);
      });
    };

    const advanceTimer = setInterval(() => {
      elapsed += 500;
      const cumulativeTime = stageTimings.slice(0, currentStage + 1).reduce((a, b) => a + b, 0);

      if (elapsed >= cumulativeTime && currentStage < stageTimings.length - 1) {
        currentStage++;
        advanceStage(currentStage);
        const stageLabel = INITIAL_STAGES[currentStage].label;
        addLog('info', `Pipeline stage: ${stageLabel}`);

        if (currentStage === 3) activateModels();
      }
    }, 500);

    // Poll for results
    const pollTimer = setInterval(async () => {
      try {
        const res = await apiService.getAnalysisResults(jid);
        clearInterval(pollTimer);
        clearInterval(advanceTimer);

        // Mark all stages as done
        setStages(prev => prev.map(s => ({ ...s, status: 'done' })));

        // Update models from results
        if (res.classification_details?.per_model_predictions) {
          const preds = res.classification_details.per_model_predictions as Record<string, {
            predicted_class: string; confidence: number;
          }>;
          setModels(prev => prev.map(m => {
            const pred = preds[m.name];
            return pred ? {
              ...m,
              predictedClass: pred.predicted_class,
              confidence: pred.confidence,
              status: 'done',
            } : { ...m, status: 'done' };
          }));
        }

        // Agreement score
        const agreement = (res.classification_details?.agreement_score as number) ?? 1;
        setAgreementScore(agreement);

        setResult(res);
        setSystemStatus(s => ({
          ...s,
          lastInference: new Date().toLocaleTimeString(),
          casesAnalyzed: s.casesAnalyzed + 1,
        }));
        addLog('success', `Analysis complete — ${res.tumor_detected ? res.tumor_type || 'Tumor detected' : 'No tumor detected'} (${(res.confidence * 100).toFixed(1)}%)`);

        // Transition to workspace after brief delay
        setTimeout(() => setScreen('workspace'), 600);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '';
        if (msg.includes('still in progress')) return; // keep polling
        // Real error
        clearInterval(pollTimer);
        clearInterval(advanceTimer);
        setError(msg || 'Analysis failed');
        addLog('error', `Analysis failed: ${msg}`);
      }
    }, 2000);

    pollingRef.current = pollTimer;

    // Try WebSocket for real-time updates
    try {
      const ws = apiService.subscribeToJobStatus(jid, (msg) => {
        if (msg.status) addLog('info', `WS: ${msg.event || msg.status}`);
      });
      wsRef.current = ws;
    } catch { /* WS optional */ }
  }, [addLog, advanceStage]);

  /* ── Upload handler ──────────────────────────────────────── */
  const handleUpload = useCallback(async (file: File) => {
    setSelectedFile(file);
    setError(null);
    setResult(null);
    setStages(INITIAL_STAGES);
    setModels(INITIAL_MODELS);
    setAgreementScore(0);

    // Preview
    const reader = new FileReader();
    reader.onload = e => setFilePreview(e.target?.result as string);
    reader.readAsDataURL(file);

    addLog('info', `File selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

    // Start upload after brief validation animation
    setTimeout(async () => {
      try {
        addLog('info', 'Uploading image to server…');
        const response = await apiService.analyzeImage(file, (progress) => {
          setUploadProgress(progress);
        });

        setJobId(response.job_id);
        setUploadProgress(100);
        addLog('success', `Upload complete — Job ID: ${response.job_id.substring(0, 8)}…`);

        // Transition to pipeline view
        setScreen('pipeline');
        startPipelineTracking(response.job_id);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        setError(msg);
        addLog('error', msg);
      }
    }, 1500);
  }, [addLog, startPipelineTracking]);

  /* ── Navigation handler ──────────────────────────────────── */
  const handleNav = useCallback((target: string) => {
    if (target === 'command' || target === 'ingestion') {
      setScreen(target as AppScreen);
    } else if (target === 'workspace' && result) {
      setScreen('workspace');
    }
  }, [result]);

  const startNewAnalysis = useCallback(() => {
    setScreen('ingestion');
    setSelectedFile(null);
    setFilePreview(null);
    setUploadProgress(0);
    setJobId(null);
    setResult(null);
    setError(null);
    setStages(INITIAL_STAGES);
    setModels(INITIAL_MODELS);
    setAgreementScore(0);
    addLog('info', 'Starting new analysis session');
  }, [addLog]);

  /* ── Render ──────────────────────────────────────────────── */
  return (
    <div className={`crw-root ${bottomOpen ? '' : 'bottom-closed'}`}>
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="crw-header">
        <div className="crw-header-left">
          <div className="crw-logo">
            <div className="crw-logo-icon">◈</div>
            <span>AI Masters</span>
          </div>
          <div className="crw-status-chip">
            <span className={`crw-status-dot ${systemStatus.systemHealth === 'online' ? '' : systemStatus.systemHealth === 'degraded' ? 'warning' : 'error'}`} />
            {systemStatus.systemHealth === 'online' ? 'All Systems Operational' : 'System Degraded'}
          </div>
          <div className="crw-status-chip">
            <span className="crw-status-dot" />
            {systemStatus.modelsOperational}/{systemStatus.totalModels} Models Active
          </div>
        </div>
        <div className="crw-header-right">
          {systemStatus.lastInference !== 'None' && (
            <div className="crw-status-chip">
              Last: {systemStatus.lastInference}
            </div>
          )}
          <button className="crw-theme-toggle" onClick={toggleTheme} title="Toggle theme">
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>
      </header>

      {/* ── Sidebar ─────────────────────────────────────────── */}
      <nav
        className={`crw-sidebar ${sidebarExpanded ? 'expanded' : ''}`}
        onMouseEnter={() => setSidebarExpanded(true)}
        onMouseLeave={() => setSidebarExpanded(false)}
      >
        {SIDEBAR_NAV.map(item => (
          <button
            key={item.id}
            className={`crw-sidebar-item ${screen === item.id || (item.id === 'workspace' && screen === 'pipeline') ? 'active' : ''}`}
            onClick={() => handleNav(item.id)}
            title={item.label}
          >
            {item.icon}
            <span className="crw-sidebar-label">{item.label}</span>
          </button>
        ))}
        <div className="crw-sidebar-divider" />
        {SIDEBAR_TOOLS.map(item => (
          <button
            key={item.id}
            className="crw-sidebar-item"
            title={item.label}
            onClick={() => {
              if (item.id === 'history') addLog('info', 'Case history — see analytics panel');
            }}
          >
            {item.icon}
            <span className="crw-sidebar-label">{item.label}</span>
          </button>
        ))}
      </nav>

      {/* ── Main Content ────────────────────────────────────── */}
      <main className="crw-main">
        {screen === 'command' && (
          <CommandEntry
            systemStatus={systemStatus}
            onStartAnalysis={startNewAnalysis}
          />
        )}
        {screen === 'ingestion' && (
          <SmartIngestion
            selectedFile={selectedFile}
            filePreview={filePreview}
            uploadProgress={uploadProgress}
            error={error}
            onFileSelect={handleUpload}
          />
        )}
        {screen === 'pipeline' && (
          <PipelineView
            stages={stages}
            models={models}
            agreementScore={agreementScore}
            result={result}
          />
        )}
        {screen === 'workspace' && result && (
          <DiagnosticWorkspace
            result={result}
            models={models}
            agreementScore={agreementScore}
            filePreview={filePreview}
            fileName={selectedFile?.name || ''}
          />
        )}
      </main>

      {/* ── Bottom Panel (Pipeline Logs) ────────────────────── */}
      <PipelineLog
        logs={logs}
        isOpen={bottomOpen}
        onToggle={() => setBottomOpen(o => !o)}
      />

      {/* ── AI Chat Assistant ───────────────────────────────── */}
      <AIChatAssistant result={result} models={models} agreementScore={agreementScore} />
    </div>
  );
};

export default Dashboard;
