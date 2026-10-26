import React from 'react';
import { Menu, RefreshCw, BarChart2 } from 'lucide-react';

export default function DashboardHeader({
  activeTab = 'dashboard',
  onRefresh,
  refreshing = false,
  onToggleSidebar,
}) {
  const titles = {
    dashboard: {
      title: 'Overview',
      subtitle: 'Automated image model benchmarking and selection for edge deployment.',
    },
    compare: {
      title: 'Model Comparison Matrix',
      subtitle: 'Side-by-side empirical performance evaluations across models and runtimes.',
    },
    benchmark: {
      title: 'Run Evaluation & Profiling',
      subtitle: 'Execute live local Host CPU benchmarks or analytical Edge Impulse simulations.',
    },
    recommendations: {
      title: 'Deployment Recommendations',
      subtitle: 'Multi-criteria decision analysis (MCDA) tailored to edge device constraints.',
    },
    history: {
      title: 'Run History & Sessions',
      subtitle: 'Comprehensive audit log of saved benchmark runs and granular measurements.',
    },
  };

  const current = titles[activeTab] || titles.dashboard;

  return (
    <header
      id="app-topbar"
      style={{
        background: '#ffffff',
        borderBottom: '1px solid #e2e8f0',
        padding: '1rem 1.75rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
        position: 'sticky',
        top: 0,
        zIndex: 40,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
        {/* Mobile menu trigger */}
        <button
          id="btn-mobile-menu"
          onClick={onToggleSidebar}
          aria-label="Toggle navigation menu"
          className="mobile-menu-toggle"
          style={{
            background: 'transparent',
            border: '1px solid #cbd5e1',
            borderRadius: '6px',
            color: '#334155',
            padding: '0.35rem',
            cursor: 'pointer',
            display: 'none',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Menu size={18} />
        </button>

        <div>
          <h1
            style={{
              fontSize: '1.25rem',
              fontWeight: 800,
              color: '#0f172a',
              letterSpacing: '-0.02em',
              margin: 0,
            }}
          >
            {current.title}
          </h1>
          <p
            style={{
              fontSize: '0.8rem',
              color: '#64748b',
              margin: '0.15rem 0 0',
            }}
          >
            {current.subtitle}
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <button
          id="btn-refresh-data"
          onClick={onRefresh}
          disabled={refreshing}
          className="btn btn-sm"
          style={{
            background: '#ffffff',
            border: '1px solid #cbd5e1',
            color: '#334155',
            fontSize: '0.775rem',
            fontWeight: 600,
            padding: '0.45rem 0.85rem',
            borderRadius: '6px',
            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
            cursor: refreshing ? 'not-allowed' : 'pointer',
          }}
        >
          <RefreshCw
            size={13}
            className={refreshing ? 'animate-spin' : ''}
            style={{
              animation: refreshing ? 'spin 1s linear infinite' : 'none',
              color: '#2563eb',
            }}
          />
          <span>{refreshing ? 'Syncing...' : 'Sync Repository'}</span>
        </button>
      </div>
    </header>
  );
}
