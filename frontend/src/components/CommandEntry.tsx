import React, { useState } from 'react';
import type { SystemStatus } from './Dashboard';

interface Props {
  systemStatus: SystemStatus;
  onStartAnalysis: () => void;
}

/* ── Model metadata ────────────────────────────────────────── */
const MODEL_DETAILS = [
  {
    name: 'Custom CNN',
    icon: '⊞',
    color: '#00d4ff',
    accentGlow: 'rgba(0, 212, 255, 0.25)',
    architecture: 'Custom Convolutional Neural Network',
    description:
      'A purpose-built convolutional neural network designed specifically for brain tumor MRI classification. This lightweight architecture uses depth-wise separable convolutions and residual skip connections for high-speed inference while maintaining competitive accuracy.',
    strengths: [
      'Optimized for brain MRI modalities',
      'Fastest inference time (~12 ms)',
      'Low memory footprint — runs on CPU',
      'Fine-tuned on combined Br35H + Figshare datasets',
    ],
    specs: { params: '2.1 M', layers: 18, inputSize: '224×224', accuracy: '94.2%' },
  },
  {
    name: 'ResNet-50',
    icon: '◈',
    color: '#3b82f6',
    accentGlow: 'rgba(59, 130, 246, 0.25)',
    architecture: 'Residual Network — 50 layers',
    description:
      'ResNet-50 introduced the revolutionary skip-connection paradigm that solved the vanishing-gradient problem in deep networks. Pre-trained on ImageNet and fine-tuned for brain MRI, it delivers strong feature extraction through 50 layers of residual blocks.',
    strengths: [
      'Proven architecture — millions of citations',
      'Excellent generalization across tumor types',
      'Robust to noise and imaging artifacts',
      'Deep residual learning captures subtle patterns',
    ],
    specs: { params: '25.6 M', layers: 50, inputSize: '224×224', accuracy: '96.1%' },
  },
  {
    name: 'EfficientNet-B0',
    icon: '◎',
    color: '#10b981',
    accentGlow: 'rgba(16, 185, 129, 0.25)',
    architecture: 'Compound-Scaled Network — B0 variant',
    description:
      'EfficientNet uses compound scaling to uniformly scale depth, width, and resolution with a fixed set of coefficients. The B0 variant achieves state-of-the-art accuracy with far fewer parameters than comparable models, making it ideal for medical imaging pipelines.',
    strengths: [
      'Best accuracy-to-parameter ratio',
      'Compound scaling balances all network dimensions',
      'Mobile-inverted bottleneck (MBConv) blocks',
      'Squeeze-and-excitation attention built in',
    ],
    specs: { params: '5.3 M', layers: 237, inputSize: '224×224', accuracy: '96.8%' },
  },
  {
    name: 'DenseNet-121',
    icon: '⊕',
    color: '#a855f7',
    accentGlow: 'rgba(168, 85, 247, 0.25)',
    architecture: 'Densely Connected Network — 121 layers',
    description:
      'DenseNet-121 connects every layer to every other layer in a feed-forward fashion. This dense connectivity pattern encourages feature reuse, strengthens gradient flow, and substantially reduces the number of parameters compared to traditional CNNs of similar depth.',
    strengths: [
      'Dense connectivity maximizes feature reuse',
      'Strong gradient flow through all layers',
      'Excellent at detecting small / subtle tumors',
      'Reduced parameters via feature concatenation',
    ],
    specs: { params: '8.0 M', layers: 121, inputSize: '224×224', accuracy: '95.7%' },
  },
];

const CommandEntry: React.FC<Props> = ({ systemStatus, onStartAnalysis }) => {
  const [hoveredModel, setHoveredModel] = useState<number | null>(null);
  const [expandedModel, setExpandedModel] = useState<number | null>(null);

  return (
    <div className="crw-command-entry" style={{ justifyContent: 'flex-start', paddingTop: '28px', overflowY: 'auto' }}>
      {/* ── Hero ────────────────────────────────────────────── */}
      <div className="crw-command-title">AI Masters</div>
      <div className="crw-command-subtitle" style={{ maxWidth: '600px' }}>
        Enterprise-grade brain tumor MRI diagnostic platform powered by a 4-model deep-learning
        ensemble. Each model independently analyzes every scan, then a fusion engine merges their
        predictions through weighted voting, agreement scoring, and confidence gating to deliver
        clinician-ready explainable results.
      </div>

      {/* ── System Stats ────────────────────────────────────── */}
      <div className="crw-system-grid">
        <div className="crw-system-stat">
          <span className="value">{systemStatus.modelsOperational}</span>
          <span className="label">AI Models</span>
        </div>
        <div className="crw-system-stat">
          <span className="value" style={{
            color: systemStatus.systemHealth === 'online' ? 'var(--conf-high)' : 'var(--conf-mid)'
          }}>
            {systemStatus.systemHealth === 'online' ? '●' : '◐'}
          </span>
          <span className="label">System Status</span>
        </div>
        <div className="crw-system-stat">
          <span className="value" style={{ fontSize: '0.95rem' }}>
            {systemStatus.lastTraining}
          </span>
          <span className="label">Last Training</span>
        </div>
        <div className="crw-system-stat">
          <span className="value">{systemStatus.casesAnalyzed}</span>
          <span className="label">Cases Analyzed</span>
        </div>
      </div>

      {/* ── How It Works ────────────────────────────────────── */}
      <div className="aim-section-header">How It Works</div>
      <div className="aim-how-grid">
        {[
          { step: '1', icon: '⬆', title: 'Upload MRI Scan', desc: 'Upload a brain MRI image in JPEG, PNG, or DICOM format. The system validates the image dimensions, modality, and quality before processing.' },
          { step: '2', icon: '⊡', title: 'Pre-Processing', desc: 'The image is normalized, resized to 224×224, converted to a PyTorch tensor, and augmented via test-time augmentation (horizontal flip, vertical flip, rotation).' },
          { step: '3', icon: '◈', title: 'Ensemble Inference', desc: 'Four independent neural networks analyze the scan in parallel. Each model produces a probability distribution over four classes: glioma, meningioma, pituitary, and no tumor.' },
          { step: '4', icon: '⊕', title: 'Fusion & Gating', desc: 'Predictions are merged via weighted averaging. A confidence gate evaluates agreement across models and flags low-confidence results for mandatory specialist review.' },
        ].map((item, i) => (
          <div key={i} className="aim-how-card">
            <div className="aim-how-step">{item.step}</div>
            <div className="aim-how-icon">{item.icon}</div>
            <div className="aim-how-title">{item.title}</div>
            <div className="aim-how-desc">{item.desc}</div>
          </div>
        ))}
      </div>

      {/* ── AI Models — Individual Cards ─────────────────────── */}
      <div className="aim-section-header" style={{ marginTop: '12px' }}>AI Model Ensemble</div>
      <div className="aim-section-subtitle">
        Hover over each model to preview its architecture. Click to expand full details, strengths, and specifications.
      </div>

      <div className="aim-models-grid">
        {MODEL_DETAILS.map((model, idx) => {
          const isHovered = hoveredModel === idx;
          const isExpanded = expandedModel === idx;

          return (
            <div
              key={model.name}
              className={`aim-model-box ${isHovered ? 'hovered' : ''} ${isExpanded ? 'expanded' : ''}`}
              style={{
                '--model-color': model.color,
                '--model-glow': model.accentGlow,
              } as React.CSSProperties}
              onMouseEnter={() => setHoveredModel(idx)}
              onMouseLeave={() => setHoveredModel(null)}
              onClick={() => setExpandedModel(isExpanded ? null : idx)}
            >
              {/* Top accent bar */}
              <div className="aim-model-accent" />

              {/* Icon + Name header */}
              <div className="aim-model-head">
                <div className="aim-model-icon">{model.icon}</div>
                <div>
                  <div className="aim-model-name">{model.name}</div>
                  <div className="aim-model-arch">{model.architecture}</div>
                </div>
                <div className="aim-model-status-dot" />
              </div>

              {/* Short description */}
              <div className="aim-model-desc">{model.description}</div>

              {/* Spec pills — always visible */}
              <div className="aim-model-specs">
                <span className="aim-spec-pill"><b>Params</b> {model.specs.params}</span>
                <span className="aim-spec-pill"><b>Layers</b> {model.specs.layers}</span>
                <span className="aim-spec-pill"><b>Input</b> {model.specs.inputSize}</span>
                <span className="aim-spec-pill aim-spec-accent"><b>Accuracy</b> {model.specs.accuracy}</span>
              </div>

              {/* Expanded section */}
              {isExpanded && (
                <div className="aim-model-expanded">
                  <div className="aim-strengths-title">Key Strengths</div>
                  <ul className="aim-strengths-list">
                    {model.strengths.map((s, si) => (
                      <li key={si}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Expand hint */}
              <div className="aim-model-expand-hint">
                {isExpanded ? '▲ Collapse' : '▼ Click to expand'}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Platform Features ───────────────────────────────── */}
      <div className="aim-section-header" style={{ marginTop: '12px' }}>Platform Capabilities</div>
      <div className="aim-features-grid">
        {[
          { icon: '⊞', title: 'Test-Time Augmentation', desc: 'Every scan is evaluated under multiple geometric transforms (flip, rotate, crop) to reduce variance and improve robustness.' },
          { icon: '⊕', title: 'Weighted Ensemble Fusion', desc: 'Model outputs are combined using learned importance weights, giving higher influence to models that perform best on each tumor class.' },
          { icon: '⊘', title: 'Confidence Gating', desc: 'A safety layer that flags predictions below a configurable threshold. Low-confidence results trigger mandatory specialist review.' },
          { icon: '◇', title: 'Explainable AI', desc: 'Dual-mode explanation engine: technical mode for clinicians and a patient-friendly plain-language summary. Includes heatmap and segmentation overlays.' },
          { icon: '⊶', title: 'Case Intelligence', desc: 'FalkorDB graph database stores every analyzed case, enabling similar-case retrieval, trend analysis, and knowledge graph queries.' },
          { icon: '⊟', title: 'Real-Time Pipeline', desc: 'Celery workers process scans asynchronously. WebSocket updates stream pipeline progress live to the frontend dashboard.' },
        ].map((f, i) => (
          <div key={i} className="aim-feature-card">
            <div className="aim-feature-icon">{f.icon}</div>
            <div className="aim-feature-title">{f.title}</div>
            <div className="aim-feature-desc">{f.desc}</div>
          </div>
        ))}
      </div>

      {/* ── CTA ─────────────────────────────────────────────── */}
      <button className="crw-start-btn" style={{ marginTop: '28px' }} onClick={onStartAnalysis}>
        Start New Analysis
      </button>

      <div style={{
        marginTop: '16px', marginBottom: '32px', fontSize: '0.68rem', color: 'var(--text-tertiary)',
        textAlign: 'center', maxWidth: '480px', lineHeight: 1.6
      }}>
        Upload a brain MRI to run the full 4-model ensemble pipeline.
        Results include per-model predictions, fused confidence, agreement scoring,
        explainability overlays, and clinical recommendations.
      </div>
    </div>
  );
};

export default CommandEntry;
