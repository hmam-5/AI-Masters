import React from 'react';
import type { PipelineStage, ModelPrediction } from './Dashboard';
import type { AnalysisResult } from '../types/api';

interface Props {
  stages: PipelineStage[];
  models: ModelPrediction[];
  agreementScore: number;
  result: AnalysisResult | null;
}

/* ── Confidence color helper ───────────────────────────────── */
function confColor(v: number): string {
  if (v >= 0.90) return 'var(--conf-high)';
  if (v >= 0.70) return 'var(--conf-mid)';
  return 'var(--conf-low)';
}
function confLevel(v: number): 'high' | 'mid' | 'low' {
  if (v >= 0.90) return 'high';
  if (v >= 0.70) return 'mid';
  return 'low';
}

const PipelineView: React.FC<Props> = ({ stages, models, agreementScore, result }) => {
  const allDone = stages.every(s => s.status === 'done');
  const finalConf = result?.confidence ?? 0;

  return (
    <div className="crw-pipeline">
      <div className="crw-pipeline-title">
        {allDone ? '✓ Analysis Complete' : 'AI Pipeline Active'}
      </div>
      <div className="crw-pipeline-subtitle">
        {allDone
          ? 'All pipeline stages completed successfully'
          : 'Real-time processing through the inference pipeline'}
      </div>

      {/* ── Pipeline Stage Timeline ─────────────────────────── */}
      <div className="crw-pipeline-stages">
        {stages.map((stage) => (
          <div key={stage.id} className={`crw-pipeline-stage ${stage.status}`}>
            <div className="crw-stage-node">
              {stage.status === 'done' ? '✓' : stage.status === 'active' ? '◌' : stage.icon}
            </div>
            <div className="crw-stage-content">
              <div className="crw-stage-name">{stage.label}</div>
              <div className="crw-stage-desc">{stage.description}</div>

              {/* TTA visual during augmentation stage */}
              {stage.id === 'tta' && stage.status === 'active' && (
                <div style={{
                  display: 'flex', gap: '8px', marginTop: '10px', flexWrap: 'wrap',
                  animation: 'fadeIn 0.4s ease-out'
                }}>
                  {['Original', 'H-Flip', 'V-Flip', 'Rotate'].map((variant, i) => (
                    <div key={i} style={{
                      padding: '6px 14px', fontSize: '0.88rem',
                      background: 'var(--bg-elevated)', borderRadius: '6px',
                      color: 'var(--cyan)', border: '1px solid var(--border-active)',
                      animation: `fadeIn ${0.3 + i * 0.15}s ease-out`,
                      fontWeight: 500
                    }}>
                      {variant}
                    </div>
                  ))}
                </div>
              )}

              {/* Preprocessing visual */}
              {stage.id === 'preprocessing' && stage.status === 'active' && (
                <div style={{
                  display: 'flex', gap: '8px', marginTop: '10px', flexWrap: 'wrap',
                  animation: 'fadeIn 0.4s ease-out'
                }}>
                  {['Normalize', 'Resize 224×224', 'Tensor'].map((step, i) => (
                    <div key={i} style={{
                      padding: '6px 14px', fontSize: '0.88rem',
                      background: 'var(--bg-elevated)', borderRadius: '6px',
                      color: 'var(--text-secondary)', border: '1px solid var(--border-primary)',
                      animation: `fadeIn ${0.3 + i * 0.2}s ease-out`,
                      fontWeight: 500
                    }}>
                      {step}
                    </div>
                  ))}
                </div>
              )}

              {/* Shimmer loading bar for active stages */}
              {stage.status === 'active' && (
                <div style={{
                  marginTop: '10px', height: '3px', borderRadius: '2px',
                  background: 'var(--border-primary)', overflow: 'hidden', width: '100%'
                }}>
                  <div className="pipeline-shimmer-bar" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ── Multi-Model Cards ───────────────────────────────── */}
      <div style={{ width: '100%', maxWidth: '750px', marginTop: '32px' }}>
        <div style={{
          fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)',
          marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '10px'
        }}>
          <span style={{ color: 'var(--cyan)' }}>◈</span>
          Multi-Model Parallel Inference
        </div>
        <div className="crw-model-grid">
          {models.map(model => (
            <div
              key={model.name}
              className={`crw-model-card ${model.status === 'running' ? 'active' : model.status === 'done' ? 'done' : ''}`}
            >
              <div className="model-name">{model.displayName}</div>
              <div className="model-status">
                {model.status === 'idle' ? 'Standby' :
                 model.status === 'running' ? 'Inferring…' :
                 'Complete'}
              </div>
              {model.status === 'done' && model.confidence > 0 && (
                <>
                  <div className="model-conf" style={{ color: confColor(model.confidence) }}>
                    {(model.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="model-pred">{model.predictedClass.replace('_', ' ')}</div>
                </>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── Consensus Engine + Confidence Governance ─────────── */}
      {allDone && result && (
        <>
          {/* Consensus Ring */}
          <div className="crw-consensus">
            <div style={{
              fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)',
              marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '10px'
            }}>
              <span style={{ color: 'var(--cyan)' }}>⊕</span>
              Consensus Engine
            </div>
            <div className="crw-agreement-ring">
              <svg viewBox="0 0 120 120">
                <circle className="bg-ring" cx="60" cy="60" r="50" />
                <circle
                  className="fg-ring"
                  cx="60" cy="60" r="50"
                  stroke={confColor(agreementScore)}
                  strokeDasharray={`${agreementScore * 314.16} 314.16`}
                  strokeDashoffset="0"
                />
              </svg>
              <div className="crw-agreement-value">
                <span className="pct" style={{ color: confColor(agreementScore) }}>
                  {(agreementScore * 100).toFixed(0)}%
                </span>
                <span className="lbl">Agreement</span>
              </div>
            </div>
            {agreementScore < 1 && (
              <div style={{
                marginTop: '12px', fontSize: '0.95rem', color: 'var(--conf-mid)',
                textAlign: 'center'
              }}>
                ⚠ Partial disagreement detected between models
              </div>
            )}
          </div>

          {/* Confidence Governance Banner */}
          <div className={`crw-confidence-banner ${confLevel(finalConf)}`}>
            {finalConf >= 0.90 ? (
              <>✓ High Confidence — Prediction accepted ({(finalConf * 100).toFixed(1)}%)</>
            ) : finalConf >= 0.70 ? (
              <>◐ Moderate Confidence — Additional review recommended ({(finalConf * 100).toFixed(1)}%)</>
            ) : (
              <>⚠ Review Required — AI confidence below safe threshold ({(finalConf * 100).toFixed(1)}%)</>
            )}
          </div>

          {/* Per-model confidence grid */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px',
            width: '100%', maxWidth: '750px', marginTop: '14px'
          }}>
            {models.filter(m => m.status === 'done' && m.confidence > 0).map(m => (
              <div key={m.name} style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '12px 16px',
                background: 'var(--bg-card)', border: '1px solid var(--border-primary)',
                borderRadius: 'var(--radius-sm)', fontSize: '0.95rem'
              }}>
                <span style={{ flex: 1, color: 'var(--text-secondary)', fontWeight: 500 }}>
                  {m.displayName}
                </span>
                <span style={{
                  fontWeight: 700, fontFamily: 'var(--font-mono)',
                  color: confColor(m.confidence)
                }}>
                  {(m.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '10px',
              padding: '12px 16px',
              background: 'var(--cyan-dim)', border: '1px solid var(--border-active)',
              borderRadius: 'var(--radius-sm)', fontSize: '0.95rem',
              gridColumn: '1 / -1'
            }}>
              <span style={{ flex: 1, color: 'var(--cyan)', fontWeight: 600 }}>
                Fused Ensemble Confidence
              </span>
              <span style={{
                fontWeight: 700, fontFamily: 'var(--font-mono)',
                color: confColor(finalConf), fontSize: '1.1rem'
              }}>
                {(finalConf * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default PipelineView;
