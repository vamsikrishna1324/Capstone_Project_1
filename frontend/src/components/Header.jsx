import React from 'react';
import { Activity, Database, CheckCircle, AlertTriangle, Cpu } from 'lucide-react';

export default function Header({ health, activeTab, onTabChange }) {
  const isHealthy = health?.status === 'ok';
  const isDbConnected = health?.database?.connected === true;

  const tabs = [
    { id: 'dashboard', label: 'Overview' },
    { id: 'compare', label: 'Model Comparison' },
    { id: 'benchmark', label: 'Run Benchmark' },
    { id: 'recommendations', label: 'Recommendations' },
    { id: 'history', label: 'Run History' },
  ];

  return (
    <header className="glass-panel" style={{ margin: '1.25rem 1.5rem', padding: '1.25rem 1.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.25rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                background: 'linear-gradient(135deg, var(--primary-500) 0%, var(--accent-cyan) 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 15px var(--primary-glow)',
              }}
            >
              <Cpu size={18} color="#ffffff" />
            </div>
            <h1 style={{ fontSize: '1.4rem', fontWeight: 800 }}>
              Automated Image Model Benchmarking
            </h1>
            <span className="badge badge-primary" style={{ fontSize: '0.65rem' }}>
              Edge Deployment
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '640px' }}>
            Automated comparative profiling across PyTorch, ONNX Runtime, and TensorFlow Lite with multi-criteria edge recommendations.
          </p>
        </div>

        {/* System Health Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div
            className="glass-card"
            style={{
              padding: '0.5rem 0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.8rem',
            }}
          >
            <span className={`pulsing-dot ${isHealthy ? 'pulsing-dot-green' : ''}`} style={{ backgroundColor: isHealthy ? 'var(--success)' : 'var(--warning)' }} />
            <span style={{ color: 'var(--text-muted)' }}>API:</span>
            <span style={{ fontWeight: 600, color: isHealthy ? '#6ee7b7' : '#fcd34d' }}>
              {health?.api || 'Checking...'}
            </span>
          </div>

          <div
            className="glass-card"
            style={{
              padding: '0.5rem 0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.8rem',
            }}
          >
            <Database size={14} color={isDbConnected ? 'var(--success)' : 'var(--danger)'} />
            <span style={{ color: 'var(--text-muted)' }}>MySQL:</span>
            <span style={{ fontWeight: 600, color: isDbConnected ? '#6ee7b7' : '#fca5a5' }}>
              {isDbConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav
        aria-label="Dashboard Sections"
        style={{
          display: 'flex',
          gap: '0.5rem',
          marginTop: '1.25rem',
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '1rem',
          overflowX: 'auto',
        }}
      >
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              onClick={() => onTabChange(tab.id)}
              className="btn btn-sm"
              style={{
                background: isActive ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(6, 182, 212, 0.2) 100%)' : 'transparent',
                borderColor: isActive ? 'var(--primary-500)' : 'transparent',
                color: isActive ? '#ffffff' : 'var(--text-muted)',
                fontWeight: isActive ? 600 : 500,
                boxShadow: isActive ? '0 0 12px var(--primary-glow)' : 'none',
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
