// Main React dashboard component.

import React, { useState, useCallback, useEffect, useRef } from "react";
import UploadForm from "./UploadForm";
import apiService from "../services/api";
import type { AnalysisResult, ImageAnalyzeResponse } from "../types/api";

type AppState = "upload" | "processing" | "results" | "error";

/* ─── Animated confidence ring ─── */
const ConfidenceGauge: React.FC<{ value: number; size?: number; color?: string }> = ({
  value,
  size = 120,
  color = "#6366f1",
}) => {
  const [animVal, setAnimVal] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimVal(value), 100);
    return () => clearTimeout(t);
  }, [value]);
  const r = (size - 12) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (animVal / 100) * circ;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="10" />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: size * 0.22, fontWeight: 800, color: "#fff" }}>
          {(animVal).toFixed(1)}%
        </span>
        <span style={{ fontSize: size * 0.1, color: "rgba(255,255,255,0.6)", fontWeight: 500 }}>
          confidence
        </span>
      </div>
    </div>
  );
};

/* ─── Animated counter ─── */
const AnimatedNumber: React.FC<{ value: number; decimals?: number; suffix?: string }> = ({
  value, decimals = 1, suffix = "",
}) => {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = value / 40;
    const interval = setInterval(() => {
      start += step;
      if (start >= value) { setDisplay(value); clearInterval(interval); }
      else setDisplay(start);
    }, 25);
    return () => clearInterval(interval);
  }, [value]);
  return <>{display.toFixed(decimals)}{suffix}</>;
};

/* ─── Step card for processing ─── */
const ProcessingStep: React.FC<{ label: string; active: boolean; done: boolean; icon: string }> = ({
  label, active, done, icon,
}) => (
  <div style={{
    display: "flex", alignItems: "center", gap: "14px", padding: "14px 18px",
    borderRadius: "12px", marginBottom: "8px",
    background: done ? "rgba(16,185,129,0.1)" : active ? "rgba(99,102,241,0.1)" : "rgba(255,255,255,0.04)",
    border: `1px solid ${done ? "rgba(16,185,129,0.25)" : active ? "rgba(99,102,241,0.25)" : "rgba(255,255,255,0.06)"}`,
    transition: "all 0.5s ease",
  }}>
    <div style={{
      width: 36, height: 36, borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center",
      background: done ? "rgba(16,185,129,0.15)" : active ? "rgba(99,102,241,0.15)" : "rgba(255,255,255,0.06)",
      fontSize: "16px",
    }}>
      {done ? "✓" : icon}
    </div>
    <span style={{
      fontSize: "13px", fontWeight: 500,
      color: done ? "#10b981" : active ? "#a5b4fc" : "rgba(255,255,255,0.35)",
    }}>
      {label}
    </span>
    {active && !done && (
      <div style={{ marginLeft: "auto" }}>
        <div style={{
          width: 18, height: 18, border: "2px solid rgba(99,102,241,0.3)",
          borderTop: "2px solid #818cf8", borderRadius: "50%", animation: "spin 0.8s linear infinite",
        }} />
      </div>
    )}
  </div>
);

/* ─── Fade-in wrapper ─── */
const FadeIn: React.FC<{ children: React.ReactNode; delay?: number }> = ({ children, delay = 0 }) => {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), delay); return () => clearTimeout(t); }, [delay]);
  return (
    <div style={{
      opacity: visible ? 1 : 0, transform: visible ? "translateY(0)" : "translateY(20px)",
      transition: "opacity 0.6s ease, transform 0.6s ease",
    }}>
      {children}
    </div>
  );
};

/* ═══════════════════════════════  DASHBOARD  ═══════════════════════════════ */

export const Dashboard: React.FC = () => {
  const [appState, setAppState] = useState<AppState>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollProgress, setPollProgress] = useState(0);
  const [processingStep, setProcessingStep] = useState(0);

  /* Cycle through processing steps for visual feedback */
  useEffect(() => {
    if (appState !== "processing") return;
    const steps = [0, 1, 2, 3];
    let idx = 0;
    const iv = setInterval(() => {
      idx = Math.min(idx + 1, steps.length - 1);
      setProcessingStep(idx);
    }, 2200);
    return () => clearInterval(iv);
  }, [appState]);

  const pollForResults = useCallback(async (id: string) => {
    let attempts = 0;
    const maxAttempts = 120;
    const poll = async () => {
      if (attempts >= maxAttempts) {
        setError("Analysis timed out. Please try again.");
        setAppState("error");
        return;
      }
      attempts++;
      setPollProgress(Math.min(95, attempts * 2));
      try {
        const result = await apiService.getAnalysisResults(id);
        if (result && result.findings && result.explanation) {
          setResults(result);
          setAppState("results");
        } else {
          setTimeout(poll, 2000);
        }
      } catch (err: any) {
        const msg = err?.message || "";
        if (msg.includes("still in progress") || msg.includes("202") || msg.includes("processing")) {
          setTimeout(poll, 2000);
        } else {
          setTimeout(poll, 3000);
        }
      }
    };
    poll();
  }, []);

  const handleAnalysisStarted = (response: ImageAnalyzeResponse) => {
    setJobId(response.job_id);
    setAppState("processing");
    setPollProgress(0);
    setProcessingStep(0);
    pollForResults(response.job_id);
  };

  const handleNewAnalysis = () => {
    setAppState("upload");
    setJobId(null);
    setResults(null);
    setError(null);
    setPollProgress(0);
    setProcessingStep(0);
  };

  const css = `
    @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
    @keyframes fadeIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
    @keyframes slideUp { from{opacity:0;transform:translateY(30px)} to{opacity:1;transform:translateY(0)} }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes brainPulse { 0%,100%{transform:scale(1);filter:drop-shadow(0 0 20px rgba(99,102,241,0.3))} 50%{transform:scale(1.05);filter:drop-shadow(0 0 40px rgba(139,92,246,0.5))} }
    @keyframes scanLine { 0%{top:0%;opacity:0} 10%{opacity:1} 90%{opacity:1} 100%{top:100%;opacity:0} }
    @keyframes orbitDot { 0%{transform:rotate(0deg) translateX(52px) rotate(0deg)} 100%{transform:rotate(360deg) translateX(52px) rotate(-360deg)} }
    @keyframes ripple { 0%{box-shadow:0 0 0 0 rgba(99,102,241,0.3)} 100%{box-shadow:0 0 0 20px rgba(99,102,241,0)} }
    @keyframes barGrow { from{width:0%} to{width:var(--bar-w)} }
    * { box-sizing: border-box; }
    body { margin: 0; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
    .glass-card {
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      transition: all 0.3s ease;
    }
    .glass-card:hover { border-color: rgba(255,255,255,0.18); background: rgba(255,255,255,0.08); }
    .btn-primary {
      padding: 14px 32px; border: none; border-radius: 12px; cursor: pointer;
      font-size: 15px; font-weight: 600; color: #fff;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      box-shadow: 0 4px 20px rgba(99,102,241,0.35);
      transition: all 0.3s ease;
    }
    .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(99,102,241,0.5); }
  `;

  return (
    <div style={{
      width: "100%",
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      background: "linear-gradient(135deg, #0f0b1e 0%, #1a1040 30%, #0d1933 60%, #0a0f1e 100%)",
      backgroundSize: "400% 400%",
      animation: "gradientShift 15s ease infinite",
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      color: "#e2e8f0",
    }}>
      <style>{css}</style>

      {/* ─── Background decorations ─── */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{
          position: "absolute", top: "-20%", right: "-10%", width: "600px", height: "600px",
          borderRadius: "50%", background: "radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%)",
        }} />
        <div style={{
          position: "absolute", bottom: "-15%", left: "-10%", width: "500px", height: "500px",
          borderRadius: "50%", background: "radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)",
        }} />
      </div>

      {/* ─── Header ─── */}
      <header style={{
        padding: "16px 32px",
        background: "rgba(15,11,30,0.6)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        position: "sticky", top: 0, zIndex: 50,
      }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
            <div style={{
              width: 42, height: 42, borderRadius: "12px",
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 4px 16px rgba(99,102,241,0.4)",
            }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a7 7 0 0 0-7 7c0 3 1.5 5.5 3 7.5S12 22 12 22s2.5-3.5 4-5.5 3-4.5 3-7.5a7 7 0 0 0-7-7z"/>
                <circle cx="12" cy="9" r="2.5"/>
              </svg>
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: "18px", fontWeight: 700, color: "#f1f5f9", letterSpacing: "-0.3px" }}>
                AI Masters
              </h1>
              <p style={{ margin: 0, fontSize: "12px", color: "rgba(148,163,184,0.8)", fontWeight: 400 }}>
                Advanced Brain Tumor Detection
              </p>
            </div>
          </div>
          <div style={{
            display: "flex", alignItems: "center", gap: "6px",
            padding: "6px 14px", borderRadius: "20px",
            background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)",
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 8px #10b981" }} />
            <span style={{ fontSize: "12px", color: "#10b981", fontWeight: 500 }}>System Online</span>
          </div>
        </div>
      </header>

      {/* ─── Main Content ─── */}
      <main style={{ flex: 1, padding: "40px 20px", position: "relative", zIndex: 1 }}>

        {/* ══════ UPLOAD STATE ══════ */}
        {appState === "upload" && (
          <div style={{ maxWidth: "580px", margin: "0 auto", animation: "fadeIn 0.6s ease" }}>
            {/* Hero text */}
            <div style={{ textAlign: "center", marginBottom: "32px" }}>
              <h2 style={{
                margin: "0 0 10px", fontSize: "28px", fontWeight: 800,
                background: "linear-gradient(135deg, #c7d2fe, #a5b4fc, #818cf8)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                letterSpacing: "-0.5px",
              }}>
                Analyze Brain MRI
              </h2>
              <p style={{ margin: 0, color: "#94a3b8", fontSize: "15px", maxWidth: "420px", marginLeft: "auto", marginRight: "auto", lineHeight: 1.6 }}>
                Upload your brain scan and receive AI-powered analysis with detailed findings and recommendations
              </p>
            </div>

            {/* Upload card */}
            <div className="glass-card" style={{ padding: "2px" }}>
              <UploadForm
                onAnalysisStarted={handleAnalysisStarted}
                onError={(msg) => { setError(msg); setAppState("error"); }}
              />
            </div>

            {/* Feature pills */}
            <div style={{ display: "flex", justifyContent: "center", gap: "10px", marginTop: "28px", flexWrap: "wrap" }}>
              {[
                { icon: "⚡", text: "Instant Results" },
                { icon: "🔬", text: "Deep Analysis" },
                { icon: "🛡️", text: "Secure & Private" },
              ].map((f, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: "6px",
                  padding: "8px 16px", borderRadius: "24px",
                  background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                  fontSize: "12px", color: "#94a3b8", fontWeight: 500,
                }}>
                  <span style={{ fontSize: "14px" }}>{f.icon}</span> {f.text}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══════ PROCESSING STATE ══════ */}
        {appState === "processing" && (
          <div style={{ maxWidth: "520px", margin: "0 auto", animation: "fadeIn 0.6s ease" }}>
            <div className="glass-card" style={{ padding: "48px 36px", textAlign: "center" }}>
              {/* Animated brain scanner */}
              <div style={{ position: "relative", width: 120, height: 120, margin: "0 auto 28px" }}>
                {/* Orbit ring */}
                <div style={{
                  position: "absolute", inset: 0, borderRadius: "50%",
                  border: "1px solid rgba(99,102,241,0.2)",
                }} />
                {/* Orbiting dot */}
                <div style={{
                  position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
                  animation: "spin 3s linear infinite",
                }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%", background: "#818cf8",
                    boxShadow: "0 0 12px #818cf8", position: "absolute", top: 0, left: "50%", marginLeft: -4,
                  }} />
                </div>
                {/* Center brain icon */}
                <div style={{
                  position: "absolute", inset: "20px", borderRadius: "50%",
                  background: "rgba(99,102,241,0.1)", display: "flex", alignItems: "center", justifyContent: "center",
                  animation: "brainPulse 2s ease-in-out infinite",
                }}>
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#a5b4fc" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a7 7 0 0 0-7 7c0 3 1.5 5.5 3 7.5S12 22 12 22s2.5-3.5 4-5.5 3-4.5 3-7.5a7 7 0 0 0-7-7z"/>
                    <circle cx="12" cy="9" r="2.5"/>
                  </svg>
                </div>
                {/* Ripple effect */}
                <div style={{
                  position: "absolute", inset: "20px", borderRadius: "50%",
                  animation: "ripple 2s ease-out infinite",
                }} />
              </div>

              <h2 style={{ margin: "0 0 8px", fontSize: "20px", fontWeight: 700, color: "#f1f5f9" }}>
                Analyzing Your Scan
              </h2>
              <p style={{ margin: "0 0 32px", color: "#94a3b8", fontSize: "14px" }}>
                AI models are processing the image
              </p>

              {/* Progress bar */}
              <div style={{
                width: "100%", height: "4px", borderRadius: "2px",
                background: "rgba(255,255,255,0.06)", marginBottom: "28px", overflow: "hidden",
              }}>
                <div style={{
                  height: "100%", width: `${pollProgress}%`,
                  background: "linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa)",
                  borderRadius: "2px", transition: "width 0.5s ease",
                }} />
              </div>

              {/* Steps */}
              <div style={{ textAlign: "left" }}>
                {[
                  { icon: "📤", label: "Image uploaded successfully" },
                  { icon: "🔍", label: "Running preprocessing pipeline" },
                  { icon: "🧠", label: "Neural network classification" },
                  { icon: "📊", label: "Generating analysis report" },
                ].map((s, i) => (
                  <ProcessingStep key={i} icon={s.icon} label={s.label} active={processingStep === i} done={processingStep > i} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══════ RESULTS STATE ══════ */}
        {appState === "results" && results && (
          <div style={{ maxWidth: "900px", margin: "0 auto" }}>
            {/* Hero result banner */}
            <FadeIn>
              <div className="glass-card" style={{
                padding: "32px",
                marginBottom: "20px",
                background: results.tumor_detected
                  ? "linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%)"
                  : "linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.06) 100%)",
                borderColor: results.tumor_detected ? "rgba(239,68,68,0.2)" : "rgba(16,185,129,0.2)",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "24px", flexWrap: "wrap" }}>
                  <ConfidenceGauge
                    value={results.confidence * 100}
                    color={results.tumor_detected ? "#ef4444" : "#10b981"}
                    size={110}
                  />
                  <div style={{ flex: 1, minWidth: "200px" }}>
                    <div style={{
                      display: "inline-flex", padding: "4px 12px", borderRadius: "20px", marginBottom: "10px",
                      background: results.tumor_detected ? "rgba(239,68,68,0.15)" : "rgba(16,185,129,0.15)",
                      border: `1px solid ${results.tumor_detected ? "rgba(239,68,68,0.3)" : "rgba(16,185,129,0.3)"}`,
                      fontSize: "12px", fontWeight: 600,
                      color: results.tumor_detected ? "#fca5a5" : "#6ee7b7",
                    }}>
                      {results.tumor_detected ? "ABNORMALITY DETECTED" : "NORMAL SCAN"}
                    </div>
                    <h2 style={{ margin: "0 0 6px", fontSize: "22px", fontWeight: 700, color: "#f1f5f9" }}>
                      {results.tumor_detected
                        ? `${results.tumor_type || "Tumor"} — ${results.tumor_grade}`
                        : "No Significant Abnormalities"
                      }
                    </h2>
                    <p style={{ margin: 0, color: "#94a3b8", fontSize: "13px" }}>
                      File: {results.image_filename} &nbsp;|&nbsp; Analysis completed
                    </p>
                  </div>
                </div>
              </div>
            </FadeIn>

            {/* Two-column layout for explanation + findings */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
              {/* Explanation */}
              <FadeIn delay={150}>
                <div className="glass-card" style={{ padding: "24px", height: "100%" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "8px",
                      background: "rgba(99,102,241,0.15)", display: "flex",
                      alignItems: "center", justifyContent: "center", fontSize: "15px",
                    }}>📋</div>
                    <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 600, color: "#e2e8f0" }}>
                      Explanation
                    </h3>
                  </div>
                  <p style={{ color: "#94a3b8", lineHeight: 1.75, fontSize: "13.5px", margin: 0 }}>
                    {results.explanation}
                  </p>
                </div>
              </FadeIn>

              {/* Findings */}
              <FadeIn delay={300}>
                <div className="glass-card" style={{ padding: "24px", height: "100%" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "8px",
                      background: "rgba(245,158,11,0.15)", display: "flex",
                      alignItems: "center", justifyContent: "center", fontSize: "15px",
                    }}>🔍</div>
                    <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 600, color: "#e2e8f0" }}>
                      Key Findings
                    </h3>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {results.findings.map((f, i) => (
                      <div key={i} style={{
                        display: "flex", alignItems: "flex-start", gap: "10px",
                        padding: "10px 14px", borderRadius: "10px",
                        background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                      }}>
                        <div style={{
                          width: 22, height: 22, borderRadius: "6px", flexShrink: 0,
                          background: "rgba(99,102,241,0.15)", display: "flex",
                          alignItems: "center", justifyContent: "center",
                          fontSize: "11px", color: "#a5b4fc", fontWeight: 700, marginTop: 1,
                        }}>{i + 1}</div>
                        <span style={{ fontSize: "13px", color: "#cbd5e1", lineHeight: 1.5 }}>{f}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </FadeIn>
            </div>

            {/* Segmentation */}
            {results.segmentation_summary && Object.keys(results.segmentation_summary).length > 0 && (
              <FadeIn delay={450}>
                <div className="glass-card" style={{ padding: "24px", marginBottom: "20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "8px",
                      background: "rgba(139,92,246,0.15)", display: "flex",
                      alignItems: "center", justifyContent: "center", fontSize: "15px",
                    }}>🧩</div>
                    <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 600, color: "#e2e8f0" }}>
                      Segmentation Analysis
                    </h3>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "14px" }}>
                    {Object.entries(results.segmentation_summary).map(([region, data], idx) => {
                      const colors = ["#6366f1", "#f59e0b", "#ef4444"];
                      const c = colors[idx % colors.length];
                      return (
                        <div key={region} style={{
                          padding: "18px", borderRadius: "12px",
                          background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)",
                        }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "14px" }}>
                            <div style={{ width: 10, height: 10, borderRadius: "3px", background: c }} />
                            <span style={{ fontSize: "13px", fontWeight: 600, color: "#e2e8f0" }}>
                              {region.replace(/_/g, " ").replace(/\b\w/g, ch => ch.toUpperCase())}
                            </span>
                          </div>
                          {/* Confidence bar */}
                          <div style={{ marginBottom: "10px" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px" }}>
                              <span style={{ fontSize: "11px", color: "#94a3b8" }}>Confidence</span>
                              <span style={{ fontSize: "11px", color: c, fontWeight: 600 }}>
                                <AnimatedNumber value={data.confidence * 100} suffix="%" />
                              </span>
                            </div>
                            <div style={{ width: "100%", height: "4px", borderRadius: "2px", background: "rgba(255,255,255,0.06)" }}>
                              <div style={{
                                height: "100%", borderRadius: "2px", background: c,
                                width: `${data.confidence * 100}%`,
                                transition: "width 1.5s cubic-bezier(0.4,0,0.2,1)",
                              }} />
                            </div>
                          </div>
                          {/* Volume */}
                          <div style={{ display: "flex", justifyContent: "space-between" }}>
                            <span style={{ fontSize: "11px", color: "#94a3b8" }}>Volume</span>
                            <span style={{ fontSize: "12px", color: "#e2e8f0", fontWeight: 600 }}>
                              <AnimatedNumber value={data.volume_mm3} suffix=" mm³" />
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </FadeIn>
            )}

            {/* Recommendations */}
            <FadeIn delay={600}>
              <div className="glass-card" style={{ padding: "24px", marginBottom: "20px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: "8px",
                    background: "rgba(16,185,129,0.15)", display: "flex",
                    alignItems: "center", justifyContent: "center", fontSize: "15px",
                  }}>💡</div>
                  <h3 style={{ margin: 0, fontSize: "15px", fontWeight: 600, color: "#e2e8f0" }}>
                    Recommendations
                  </h3>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "10px" }}>
                  {results.recommendations.map((rec, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "flex-start", gap: "12px",
                      padding: "12px 16px", borderRadius: "10px",
                      background: "rgba(16,185,129,0.04)",
                      border: "1px solid rgba(16,185,129,0.1)",
                    }}>
                      <div style={{
                        width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
                        background: "rgba(16,185,129,0.15)", display: "flex",
                        alignItems: "center", justifyContent: "center",
                        marginTop: 2,
                      }}>
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="3">
                          <polyline points="20 6 9 17 4 12"/>
                        </svg>
                      </div>
                      <span style={{ fontSize: "13px", color: "#cbd5e1", lineHeight: 1.5 }}>{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            </FadeIn>

            {/* Disclaimer */}
            <FadeIn delay={750}>
              <div style={{
                padding: "16px 20px", borderRadius: "12px", marginBottom: "24px",
                background: "rgba(245,158,11,0.06)", border: "1px solid rgba(245,158,11,0.15)",
                display: "flex", alignItems: "flex-start", gap: "12px",
              }}>
                <span style={{ fontSize: "18px", marginTop: "1px" }}>⚠️</span>
                <p style={{ margin: 0, fontSize: "12px", color: "#fbbf24", lineHeight: 1.7 }}>
                  <strong>Disclaimer:</strong> This analysis is generated by an AI model for educational
                  and research purposes only. It should not replace professional medical advice,
                  diagnosis, or treatment. Always consult a qualified healthcare provider.
                </p>
              </div>
            </FadeIn>

            {/* New Analysis Button */}
            <FadeIn delay={850}>
              <div style={{ textAlign: "center", paddingBottom: "30px" }}>
                <button className="btn-primary" onClick={handleNewAnalysis}>
                  <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                    </svg>
                    Analyze Another Image
                  </span>
                </button>
              </div>
            </FadeIn>
          </div>
        )}

        {/* ══════ ERROR STATE ══════ */}
        {appState === "error" && (
          <div style={{ maxWidth: "460px", margin: "60px auto", animation: "fadeIn 0.6s ease" }}>
            <div className="glass-card" style={{ padding: "48px 36px", textAlign: "center" }}>
              <div style={{
                width: 72, height: 72, borderRadius: "50%", margin: "0 auto 20px",
                background: "rgba(239,68,68,0.1)", display: "flex",
                alignItems: "center", justifyContent: "center",
              }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
              </div>
              <h2 style={{ color: "#fca5a5", margin: "0 0 8px", fontSize: "20px", fontWeight: 700 }}>
                Analysis Failed
              </h2>
              <p style={{ color: "#94a3b8", margin: "0 0 28px", fontSize: "14px", lineHeight: 1.6 }}>
                {error || "An unexpected error occurred. Please try again."}
              </p>
              <button className="btn-primary" onClick={handleNewAnalysis}>
                Try Again
              </button>
            </div>
          </div>
        )}
      </main>

      {/* ─── Footer ─── */}
      <footer style={{
        padding: "14px 20px",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        textAlign: "center", position: "relative", zIndex: 1,
      }}>
        <p style={{ margin: 0, fontSize: "11px", color: "rgba(148,163,184,0.5)" }}>
          AI Masters v2.0 &nbsp;·&nbsp; For Research & Educational Purposes Only
        </p>
      </footer>
    </div>
  );
};

export default Dashboard;
