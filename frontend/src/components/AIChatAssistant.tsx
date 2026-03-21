import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { AnalysisResult } from '../types/api';
import type { ModelPrediction } from './Dashboard';

interface Props {
  result: AnalysisResult | null;
  models: ModelPrediction[];
  agreementScore: number;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

const SUGGESTIONS = [
  'Why is there disagreement?',
  'Which model is most confident?',
  'What does the confidence mean?',
  'Explain the diagnosis',
  'What should I do next?',
];

const AIChatAssistant: React.FC<Props> = ({ result, models, agreementScore }) => {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', text: 'Hello! I\'m Supremo, your AI diagnostic companion. Ask me anything about your analysis results, model predictions, or the diagnostic process.' },
  ]);
  const [input, setInput] = useState('');
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const generateResponse = useCallback((question: string): string => {
    const q = question.toLowerCase();

    // No result yet
    if (!result) {
      return 'No analysis results are available yet. Please upload an MRI scan and run the analysis first.';
    }

    const doneModels = models.filter(m => m.status === 'done' && m.confidence > 0);
    const mostConfident = doneModels.reduce((a, b) => a.confidence > b.confidence ? a : b, doneModels[0]);
    const clf = result.classification_details || {};

    // Disagreement
    if (q.includes('disagree') || q.includes('agreement')) {
      if (agreementScore >= 1) {
        return `All ${doneModels.length} models fully agree on the prediction "${result.tumor_type || 'No tumor'}". Agreement score is 100%, meaning there is strong consensus across all architectures.`;
      }
      const disagreeModels = doneModels.filter(m => m.predictedClass !== (clf.raw_predicted_class || clf.predicted_class));
      return `The agreement score is ${(agreementScore * 100).toFixed(0)}%. ${
        disagreeModels.length > 0
          ? `${disagreeModels.map(m => m.displayName).join(', ')} predicted differently from the majority.`
          : 'Slight variations in confidence exist between models.'
      } This can happen when the case has features that are ambiguous between tumor classes. The ensemble fusion accounts for this by weighting more confident models higher.`;
    }

    // Most confident
    if (q.includes('most confident') || q.includes('which model')) {
      if (mostConfident) {
        return `${mostConfident.displayName} is the most confident model with ${(mostConfident.confidence * 100).toFixed(1)}% confidence, predicting "${mostConfident.predictedClass.replace('_', ' ')}". ${
          doneModels.length > 1
            ? `The ensemble combines all ${doneModels.length} models using weighted fusion to produce the final ${(result.confidence * 100).toFixed(1)}% confidence.`
            : ''
        }`;
      }
      return 'Model prediction data is not available yet.';
    }

    // Confidence
    if (q.includes('confidence') || q.includes('how sure') || q.includes('certain')) {
      const conf = result.confidence;
      const level = conf >= 0.90 ? 'high' : conf >= 0.70 ? 'moderate' : 'low';
      return `The fused ensemble confidence is ${(conf * 100).toFixed(1)}%, which is considered ${level}. ${
        conf >= 0.90
          ? 'This means the AI system is very confident in its prediction. All safety thresholds are met.'
          : conf >= 0.70
            ? 'Additional clinical review is recommended to confirm the finding.'
            : 'This is below the safe auto-decision threshold. Specialist review is required before any clinical decisions.'
      } The confidence is calculated by combining individual model predictions through weighted fusion with TTA (test-time augmentation) for robustness.`;
    }

    // Diagnosis
    if (q.includes('diagnos') || q.includes('tumor') || q.includes('result') || q.includes('finding')) {
      if (result.tumor_detected) {
        return `The analysis detected ${result.tumor_type || 'a brain tumor'}${result.tumor_grade ? ` classified as ${result.tumor_grade}` : ''} with ${(result.confidence * 100).toFixed(1)}% confidence. ${
          result.findings.length > 0 ? 'Key findings: ' + result.findings.join('. ') : ''
        }`;
      }
      return `No tumor was detected in this scan. The AI analyzed the image with ${(result.confidence * 100).toFixed(1)}% confidence. ${result.findings[0] || ''}`;
    }

    // Recommendations / next steps
    if (q.includes('next') || q.includes('recommend') || q.includes('should') || q.includes('do')) {
      return result.recommendations.length > 0
        ? `Based on the analysis, here are the recommendations:\n\n${result.recommendations.map((r, i) => `${i + 1}. ${r}`).join('\n')}\n\nRemember: This is AI-assisted analysis. Always consult a qualified medical professional for clinical decisions.`
        : 'Please consult with your healthcare provider for personalized medical advice based on these results.';
    }

    // Models
    if (q.includes('model') || q.includes('how many') || q.includes('architecture')) {
      return `The CRW uses 4 AI models in an ensemble:\n\n${doneModels.map(m =>
        `• ${m.displayName}: ${m.predictedClass.replace('_', ' ')} at ${(m.confidence * 100).toFixed(1)}%`
      ).join('\n')}\n\nThey work together through weighted fusion, where each model's prediction is combined based on its confidence and reliability to produce a more accurate final diagnosis.`;
    }

    // TTA
    if (q.includes('tta') || q.includes('augment')) {
      return `Test-Time Augmentation (TTA) creates ${(clf.tta_variants as number) || 4} variants of your scan (original, horizontal flip, vertical flip, rotation) and runs inference on each. The predictions are averaged to improve robustness and reduce the impact of imaging artifacts or orientation sensitivity.`;
    }

    // Default
    return `Based on the current analysis: The ${result.tumor_detected ? result.tumor_type || 'detected pathology' : 'scan'} was evaluated by ${doneModels.length} AI models with ${(agreementScore * 100).toFixed(0)}% agreement and ${(result.confidence * 100).toFixed(1)}% fused confidence. Feel free to ask about specific aspects like model disagreement, confidence levels, recommendations, or the diagnostic process.`;
  }, [result, models, agreementScore]);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed) return;
    setMessages(prev => [...prev, { role: 'user', text: trimmed }]);
    setInput('');
    // Generate response after brief delay for realism
    setTimeout(() => {
      setMessages(prev => [...prev, { role: 'assistant', text: generateResponse(trimmed) }]);
    }, 300);
  }, [input, generateResponse]);

  const handleChip = useCallback((text: string) => {
    setMessages(prev => [...prev, { role: 'user', text }]);
    setTimeout(() => {
      setMessages(prev => [...prev, { role: 'assistant', text: generateResponse(text) }]);
    }, 300);
  }, [generateResponse]);

  if (!open) {
    return (
      <button className="crw-chat-fab" onClick={() => setOpen(true)} title="Supremo">
        ◇
      </button>
    );
  }

  return (
    <div className="crw-chat-container">
      <div className="crw-chat-header">
        <h4><span style={{ color: 'var(--cyan)' }}>◇</span> Supremo</h4>
        <button className="crw-chat-close" onClick={() => setOpen(false)}>✕</button>
      </div>
      <div className="crw-chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`crw-chat-msg ${msg.role}`}>
            {msg.text.split('\n').map((line, j) => (
              <React.Fragment key={j}>
                {line}
                {j < msg.text.split('\n').length - 1 && <br />}
              </React.Fragment>
            ))}
          </div>
        ))}
        <div ref={messagesEnd} />
      </div>
      <div className="crw-chat-suggestions">
        {SUGGESTIONS.slice(0, 3).map(s => (
          <button key={s} className="crw-chat-chip" onClick={() => handleChip(s)}>{s}</button>
        ))}
      </div>
      <div className="crw-chat-input-row">
        <input
          className="crw-chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask about the analysis…"
        />
        <button className="crw-chat-send" onClick={handleSend}>Send</button>
      </div>
    </div>
  );
};

export default AIChatAssistant;
