import React, { useRef, useEffect } from 'react';
import type { LogEntry } from './Dashboard';

interface Props {
  logs: LogEntry[];
  isOpen: boolean;
  onToggle: () => void;
}

const PipelineLog: React.FC<Props> = ({ logs, isOpen, onToggle }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className={`crw-bottom ${isOpen ? 'open' : 'closed'}`}>
      <div className="crw-bottom-header" onClick={onToggle}>
        <div className="crw-bottom-tabs">
          <button className="crw-bottom-tab active">Pipeline Logs</button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
            {logs.length} entries
          </span>
          <button className="crw-bottom-toggle">
            {isOpen ? '▾' : '▴'}
          </button>
        </div>
      </div>
      {isOpen && (
        <div className="crw-log-content" ref={scrollRef}>
          {logs.length === 0 ? (
            <div style={{ color: 'var(--text-tertiary)', padding: '8px 0' }}>
              Waiting for activity…
            </div>
          ) : (
            logs.map((entry, i) => (
              <div key={i} className="crw-log-entry">
                <span className="crw-log-time">{entry.time}</span>
                <span className={`crw-log-level ${entry.level}`}>
                  {entry.level.toUpperCase()}
                </span>
                <span className="crw-log-msg">{entry.message}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default PipelineLog;
