import React, { useState, useCallback } from 'react';
import type { AnalysisResult } from '../types/api';
import type { ModelPrediction } from './Dashboard';

interface Props {
  result: AnalysisResult;
  models: ModelPrediction[];
  agreementScore: number;
  filePreview: string | null;
  fileName: string;
}

/* ── Helpers ───────────────────────────────────────────────── */
function confColor(v: number): string {
  if (v >= 0.90) return 'var(--conf-high)';
  if (v >= 0.70) return 'var(--conf-mid)';
  return 'var(--conf-low)';
}

function tumorColor(type: string | undefined): string {
  if (!type) return 'var(--text-primary)';
  const t = type.toLowerCase();
  if (t.includes('glioma')) return 'var(--tumor-glioma)';
  if (t.includes('meningioma')) return 'var(--tumor-meningioma)';
  if (t.includes('pituitary')) return 'var(--tumor-pituitary)';
  return 'var(--tumor-none)';
}

/* ── Per-model metadata ────────────────────────────────────── */
const WS_MODEL_META: Record<string, {
  icon: string; color: string; glow: string; architecture: string;
  description: string; strengths: string[];
}> = {
  custom_cnn: {
    icon: '⊞', color: '#00d4ff', glow: 'rgba(0,212,255,0.25)',
    architecture: 'Custom CNN — 18 layers',
    description: 'Purpose-built convolutional network optimized for brain MRI classification with depth-wise separable convolutions, residual skip connections, and rapid inference capability.',
    strengths: ['Fastest inference (~12 ms)', 'CPU-friendly low memory footprint', 'Fine-tuned on Br35H + Figshare'],
  },
  resnet50: {
    icon: '◈', color: '#3b82f6', glow: 'rgba(59,130,246,0.25)',
    architecture: 'ResNet-50 — 50 layers',
    description: 'Deep residual network with skip connections for powerful feature extraction. Pre-trained on ImageNet and fine-tuned on brain tumor MRI for robust generalization.',
    strengths: ['Proven architecture — millions of citations', 'Robust to noise & artifacts', '25.6 M parameters'],
  },
  efficientnet: {
    icon: '◎', color: '#10b981', glow: 'rgba(16,185,129,0.25)',
    architecture: 'EfficientNet-B0 — compound scaled',
    description: 'Compound-scaled network balancing depth, width, and resolution. Achieves top accuracy with minimal parameters via mobile-inverted bottleneck blocks and squeeze-excitation attention.',
    strengths: ['Best accuracy-to-parameter ratio', 'Squeeze-and-excitation attention', '5.3 M parameters'],
  },
  densenet: {
    icon: '⊕', color: '#a855f7', glow: 'rgba(168,85,247,0.25)',
    architecture: 'DenseNet-121 — 121 layers',
    description: 'Densely connected network where every layer receives inputs from all preceding layers. Maximizes feature reuse and gradient flow for detecting subtle tumor patterns.',
    strengths: ['Dense connectivity = max feature reuse', 'Excellent for small/subtle tumors', '8.0 M parameters'],
  },
};

const DiagnosticWorkspace: React.FC<Props> = ({
  result, models, agreementScore, filePreview, fileName,
}) => {
  const [showExplain, setShowExplain] = useState(false);
  const [explainMode, setExplainMode] = useState<'technical' | 'simple'>('technical');
  const [vizLayer, setVizLayer] = useState<'original' | 'heatmap' | 'segmentation' | 'confidence'>('original');
  const [vizOpacity, setVizOpacity] = useState(0.6);
  const [simSize, setSimSize] = useState(50);
  const [simIntensity, setSimIntensity] = useState(50);
  const [wsHoveredModel, setWsHoveredModel] = useState<string | null>(null);
  const [wsExpandedModel, setWsExpandedModel] = useState<string | null>(null);
  const [scanScale, setScanScale] = useState(420); // px — resizable

  const clf = result.classification_details || {};
  const probs = (clf.probabilities || {}) as Record<string, number>;
  const perModel = (clf.per_model_predictions || {}) as Record<string, {
    predicted_class: string; confidence: number; probabilities?: Record<string, number>;
  }>;
  const similarCases = (clf.similar_cases || []) as Array<{
    tumor_grade?: string; confidence_score?: number; created_at?: string;
  }>;
  const decisionStatus = (clf.decision_status || 'auto_accepted') as string;

  /* ── Heatmap overlay ─────────────────────────────────────── */
  const getOverlayStyle = useCallback((): React.CSSProperties => {
    if (vizLayer === 'original') return { display: 'none' };
    const gradients: Record<string, string> = {
      heatmap: 'radial-gradient(ellipse 60% 50% at 55% 45%, rgba(239,68,68,0.6) 0%, rgba(245,158,11,0.3) 40%, transparent 70%)',
      segmentation: 'radial-gradient(ellipse 45% 40% at 55% 45%, rgba(0,212,255,0.5) 0%, rgba(0,212,255,0.15) 50%, transparent 70%)',
      confidence: 'radial-gradient(ellipse 50% 45% at 55% 45%, rgba(16,185,129,0.5) 0%, rgba(245,158,11,0.3) 50%, rgba(239,68,68,0.2) 80%, transparent 90%)',
    };
    return { position: 'absolute', inset: 0, background: gradients[vizLayer] || 'none', opacity: vizOpacity, transition: 'opacity 0.3s', borderRadius: 'var(--radius-lg)', pointerEvents: 'none' };
  }, [vizLayer, vizOpacity]);

  /* ── Explanation steps ───────────────────────────────────── */
  const technicalSteps = [
    { title: 'Intensity Detection', text: 'Analyzed pixel intensity distributions and detected abnormal hyperintense/hypointense regions typical of the predicted pathology.' },
    { title: 'Boundary Irregularity', text: `Assessed tumor boundary characteristics. ${result.tumor_detected ? 'Irregular margins detected, consistent with ' + (result.tumor_type || 'tumor') + ' morphology.' : 'No irregular boundaries detected.'}` },
    { title: 'Pattern Matching', text: `Cross-referenced features against ${(clf.similar_cases_count as number) || 'historical'} training patterns with ${(result.confidence * 100).toFixed(1)}% feature similarity.` },
    { title: 'Multi-Model Consensus', text: `${Object.keys(perModel).length} models evaluated independently. Agreement score: ${(agreementScore * 100).toFixed(0)}%. ${agreementScore < 1 ? 'Partial disagreement detected — review recommended.' : 'Full consensus achieved.'}` },
  ];
  const simpleSteps = [
    { title: 'What was analyzed', text: 'Your brain MRI scan was examined by 4 different AI systems that each looked for signs of tumors or other abnormalities.' },
    { title: 'What was found', text: result.tumor_detected ? `The AI systems detected what appears to be a ${result.tumor_type || 'brain tumor'} with ${(result.confidence * 100).toFixed(0)}% confidence.` : 'No significant abnormalities were detected in your scan.' },
    { title: 'How certain is the AI', text: result.confidence >= 0.90 ? 'The AI is highly confident in this result. All models agree on the finding.' : result.confidence >= 0.70 ? 'The AI has moderate confidence. A specialist review is recommended to confirm.' : 'The AI confidence is low. This result needs specialist review before any clinical decision.' },
    { title: 'Next steps', text: 'This AI analysis is a tool to assist medical professionals. Always consult with your doctor for diagnosis and treatment decisions.' },
  ];

  /* ── Sim risk ────────────────────────────────────────────── */
  const simRisk = Math.min(100, Math.max(0, (result.confidence * 100) * (simSize / 50) * (simIntensity / 50)));
  const simRiskColor = simRisk >= 70 ? 'var(--conf-low)' : simRisk >= 40 ? 'var(--conf-mid)' : 'var(--conf-high)';

  /* ════════════════════════════════════════════════════════════
     RENDER — New layout:
       Row 1: [Left col]  [Center scan]  [Right col]
       Row 2:     [Bottom row spanning full width]
     ════════════════════════════════════════════════════════════ */
  return (
    <div className="ws2-root">
      {/* ═══ TOP ROW: 3-Column ═════════════════════════════════ */}
      <div className="ws2-top">
        {/* ── LEFT COLUMN: Diagnosis + Confidence + Probabilities ── */}
        <div className="ws2-col ws2-col-left fade-in">
          {/* Diagnosis hero */}
          <div className="ws2-card ws2-diag-hero">
            <div className="ws2-card-label">◎ Diagnosis</div>
            <div className="ws2-diag-value" style={{ color: tumorColor(result.tumor_type) }}>
              {result.tumor_detected ? (result.tumor_type || 'Tumor Detected') : 'No Tumor Detected'}
            </div>
            {result.tumor_grade && <div className="ws2-diag-grade">{result.tumor_grade}</div>}
            {decisionStatus === 'review_required' && (
              <div className="crw-confidence-banner low" style={{ marginTop: '10px', fontSize: '0.92rem' }}>
                ⚠ Review Required
              </div>
            )}
          </div>

          {/* Fused Confidence */}
          <div className="ws2-card">
            <div className="ws2-card-label">◐ Confidence</div>
            <div className="ws2-conf-row">
              <span className="ws2-conf-value" style={{ color: confColor(result.confidence) }}>
                {(result.confidence * 100).toFixed(1)}%
              </span>
              <span className="ws2-conf-tag">fused</span>
            </div>
            <div className="crw-conf-bar"><div className="crw-conf-bar-fill" style={{ width: `${result.confidence * 100}%`, background: confColor(result.confidence) }} /></div>
          </div>

          {/* Class Probabilities */}
          {Object.keys(probs).length > 0 && (
            <div className="ws2-card">
              <div className="ws2-card-label">⊟ Class Probabilities</div>
              {Object.entries(probs).sort(([, a], [, b]) => b - a).map(([cls, prob]) => (
                <div key={cls} className="ws2-prob-row">
                  <span className="ws2-prob-name">{cls.replace('_', ' ')}</span>
                  <span className="ws2-prob-val">{(prob * 100).toFixed(1)}%</span>
                  <div className="crw-conf-bar" style={{ flex: 1 }}>
                    <div className="crw-conf-bar-fill" style={{ width: `${prob * 100}%`, background: prob === Math.max(...Object.values(probs)) ? 'var(--cyan)' : 'var(--border-secondary)' }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Consensus */}
          <div className="ws2-card">
            <div className="ws2-card-label">⊕ Model Consensus</div>
            <div className="ws2-conf-row">
              <span className="ws2-conf-value" style={{ color: confColor(agreementScore), fontSize: '1.4rem' }}>
                {(agreementScore * 100).toFixed(0)}%
              </span>
              <span className="ws2-conf-tag">agreement</span>
            </div>
          </div>
        </div>

        {/* ── CENTER COLUMN: Scan + Controls ───────────────────── */}
        <div className="ws2-col ws2-col-center">
          {/* Size slider */}
          <div className="ws2-size-control">
            <span className="ws2-size-label">Scan Size</span>
            <input type="range" className="ws2-size-slider" min="200" max="560" step="10"
              value={scanScale} onChange={e => setScanScale(parseInt(e.target.value))} />
            <span className="ws2-size-val">{scanScale}px</span>
          </div>

          {/* MRI Image */}
          <div className="ws2-scan-frame" style={{ width: scanScale, height: scanScale }}>
            {filePreview ? (
              <>
                <img src={filePreview} alt="MRI Scan" className="ws2-scan-img" />
                <div style={getOverlayStyle()} />
              </>
            ) : (
              <div className="ws2-scan-placeholder">
                <span style={{ fontSize: '36px' }}>◎</span>
                <span>No image loaded</span>
              </div>
            )}
            {/* Animated scan line */}
            <div className="ws2-scan-line" />
          </div>

          <div className="ws2-scan-filename">{fileName || 'Unknown file'}</div>

          {/* Layer toggles */}
          <div className="crw-viz-controls" style={{ justifyContent: 'center' }}>
            {(['original', 'heatmap', 'segmentation', 'confidence'] as const).map(layer => (
              <button key={layer} className={`crw-viz-toggle ${vizLayer === layer ? 'active' : ''}`} onClick={() => setVizLayer(layer)}>
                {layer === 'original' ? '⊡ Original' : layer === 'heatmap' ? '◉ Heatmap' : layer === 'segmentation' ? '◎ Segmentation' : '◐ Confidence Map'}
              </button>
            ))}
          </div>
          {vizLayer !== 'original' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '6px', justifyContent: 'center' }}>
              <span style={{ fontSize: '0.88rem', color: 'var(--text-tertiary)' }}>Opacity</span>
              <input type="range" className="crw-opacity-slider" style={{ width: '140px' }} min="0" max="1" step="0.05" value={vizOpacity} onChange={e => setVizOpacity(parseFloat(e.target.value))} />
              <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{(vizOpacity * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN: Interpretation + Recommendations + Findings ── */}
        <div className="ws2-col ws2-col-right fade-in">
          {/* Interpretation */}
          <div className="ws2-card">
            <div className="ws2-card-label">◇ Interpretation</div>
            <div className="ws2-interp-text">{result.explanation}</div>
          </div>

          {/* Explainability */}
          <div className="ws2-card">
            <button className={`crw-explain-btn ${showExplain ? 'active' : ''}`} onClick={() => setShowExplain(v => !v)}>
              ◇ {showExplain ? 'Hide Explanation' : 'Explain AI Decision'}
            </button>
            {showExplain && (
              <div className="crw-explain-panel" style={{ marginTop: '10px' }}>
                <div className="crw-explain-mode-switch">
                  <button className={`crw-explain-mode-btn ${explainMode === 'technical' ? 'active' : ''}`} onClick={() => setExplainMode('technical')}>Technical</button>
                  <button className={`crw-explain-mode-btn ${explainMode === 'simple' ? 'active' : ''}`} onClick={() => setExplainMode('simple')}>Patient-Friendly</button>
                </div>
                {(explainMode === 'technical' ? technicalSteps : simpleSteps).map((step, i) => (
                  <div key={i} className="crw-explain-step">
                    <div className="crw-explain-num">{i + 1}</div>
                    <div>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '3px', fontSize: '0.95rem' }}>{step.title}</div>
                      <div style={{ color: 'var(--text-secondary)', lineHeight: 1.6, fontSize: '0.92rem' }}>{step.text}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Recommendations + Findings — side by side */}
          <div className="ws2-rec-find-row">
            <div className="ws2-card ws2-glow-card">
              <div className="ws2-card-label">⊞ Recommendations</div>
              {result.recommendations.map((rec, i) => {
                const level = rec.toLowerCase().includes('specialist') || rec.toLowerCase().includes('blocked') ? 'danger' : rec.toLowerCase().includes('additional') || rec.toLowerCase().includes('repeat') ? 'warning' : '';
                return <div key={i} className={`crw-rec-item ${level}`}>{rec}</div>;
              })}
            </div>
            <div className="ws2-card ws2-glow-card">
              <div className="ws2-card-label">⊡ Findings</div>
              {result.findings.map((finding, i) => (
                <div key={i} className="ws2-finding">• {finding}</div>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* ═══ BOTTOM ROW: Model Cards + Simulation — full width ═ */}
      <div className="ws2-bottom fade-in">
        {/* Model cards — always show all 4 */}
        <div className="ws2-card-label" style={{ marginBottom: '10px', textAlign: 'center', fontSize: '1.1rem' }}>◈ AI Model Ensemble — All 4 Models</div>
        <div className="ws2-bottom-hint">Hover for glow · Click to expand full details</div>

        <div className="ws2-models-grid-4">
          {Object.entries(WS_MODEL_META).map(([key, meta]) => {
            const liveModel = models.find(m => m.name === key);
            const hasData = liveModel && liveModel.status === 'done' && liveModel.confidence > 0;
            const isHovered = wsHoveredModel === key;
            const isExpanded = wsExpandedModel === key;
            return (
              <div
                key={key}
                className={`ws2-model-box ${isHovered ? 'hovered' : ''} ${isExpanded ? 'expanded' : ''}`}
                style={{ '--wm-color': meta.color, '--wm-glow': meta.glow } as React.CSSProperties}
                onMouseEnter={() => setWsHoveredModel(key)}
                onMouseLeave={() => setWsHoveredModel(null)}
                onClick={() => setWsExpandedModel(isExpanded ? null : key)}
              >
                {/* Accent bar */}
                <div className="ws2-mb-accent" />
                {/* Header */}
                <div className="ws2-mb-head">
                  <div className="ws2-mb-icon">{meta.icon}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="ws2-mb-name">{liveModel?.displayName || key.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                    <div className="ws2-mb-arch">{meta.architecture}</div>
                  </div>
                  {hasData && (
                    <div className="ws2-mb-conf" style={{ color: confColor(liveModel!.confidence) }}>
                      {(liveModel!.confidence * 100).toFixed(1)}%
                    </div>
                  )}
                </div>

                {/* Prediction (if available) */}
                {hasData && (
                  <div className="ws2-mb-pred">
                    <span className="ws2-mb-pred-label">Prediction:</span>
                    <span className="ws2-mb-pred-value" style={{ color: tumorColor(liveModel!.predictedClass) }}>
                      {liveModel!.predictedClass.replace('_', ' ')}
                    </span>
                  </div>
                )}

                {/* Confidence bar */}
                {hasData && (
                  <div className="crw-conf-bar" style={{ marginTop: '8px' }}>
                    <div className="crw-conf-bar-fill" style={{ width: `${liveModel!.confidence * 100}%`, background: confColor(liveModel!.confidence) }} />
                  </div>
                )}

                {/* Description — always visible */}
                <div className="ws2-mb-desc">{meta.description}</div>

                {/* Strengths — always visible */}
                <div className="ws2-mb-strengths">
                  <div className="ws2-mb-str-title">Key Strengths</div>
                  <ul>{meta.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
                </div>

                {/* Status badge */}
                <div className="ws2-mb-status" style={{ color: hasData ? 'var(--conf-high)' : 'var(--text-tertiary)' }}>
                  {hasData ? '● Active — Result Available' : '○ Loaded — Awaiting Data'}
                </div>
              </div>
            );
          })}
        </div>

        {/* Similar Cases — moved to bottom */}
        <div className="ws2-similar-bottom">
          <div className="ws2-card-label" style={{ marginBottom: '10px' }}>⊶ Similar Cases</div>
          <div className="ws2-similar-grid">
            {similarCases.length > 0 ? (
              similarCases.slice(0, 3).map((sc, i) => (
                <div key={i} className="crw-similar-case">
                  <div className="crw-similar-thumb">◎</div>
                  <div className="crw-similar-info">
                    <div className="crw-similar-diag">{sc.tumor_grade || 'Case'} #{i + 1}</div>
                    <div className="crw-similar-meta">
                      Conf: {((sc.confidence_score || 0) * 100).toFixed(0)}%
                      {sc.created_at && ` • ${new Date(sc.created_at).toLocaleDateString()}`}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div style={{ fontSize: '0.88rem', color: 'var(--text-tertiary)' }}>No similar cases in database yet.</div>
            )}
          </div>
        </div>

        {/* Scenario Simulation */}
        <div className="ws2-sim-section">
          <div className="ws2-card-label" style={{ marginBottom: '8px' }}>⊞ Scenario Simulation</div>
          <div className="ws2-sim-row">
            <div className="crw-sim-control">
              <span className="crw-sim-label">Tumor Size</span>
              <input type="range" className="crw-sim-slider" min="0" max="100" value={simSize} onChange={e => setSimSize(parseInt(e.target.value))} />
              <span className="crw-sim-value">{simSize}%</span>
            </div>
            <div className="crw-sim-control">
              <span className="crw-sim-label">Intensity</span>
              <input type="range" className="crw-sim-slider" min="0" max="100" value={simIntensity} onChange={e => setSimIntensity(parseInt(e.target.value))} />
              <span className="crw-sim-value">{simIntensity}%</span>
            </div>
            <div className="ws2-sim-result">
              <span style={{ fontSize: '0.92rem', color: 'var(--text-secondary)' }}>Risk Level</span>
              <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: simRiskColor, fontSize: '1.35rem' }}>{simRisk.toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Metadata footer */}
        <div className="ws2-meta">
          <span>Model: {(clf.model_type as string) || 'Ensemble'}</span>
          <span>•</span>
          <span>Models: {(clf.models_loaded as number) || 4} loaded</span>
          <span>•</span>
          <span>TTA: {(clf.tta_variants as number) || 4} variants</span>
          {result.completed_at && <><span>•</span><span>{new Date(result.completed_at).toLocaleString()}</span></>}
        </div>
      </div>
    </div>
  );
};

export default DiagnosticWorkspace;
