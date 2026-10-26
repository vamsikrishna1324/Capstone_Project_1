import React from 'react';
import {
  Layers,
  PlayCircle,
  Award,
  History,
  ArrowRight,
  ChevronRight,
} from 'lucide-react';
import SummaryCards from './SummaryCards';

export default function OverviewDashboard({
  models = [],
  results = [],
  totalResults = 0,
  onNavigate,
}) {
  const steps = [
    {
      num: 1,
      title: 'Compare Models',
      description: 'Go to Model Comparison to view supported models, select runtimes, and compare key metrics.',
      color: '#2563eb',
      bg: '#eff6ff',
      border: '#bfdbfe',
      targetTab: 'compare',
    },
    {
      num: 2,
      title: 'Run Benchmark',
      description: 'Navigate to Run Benchmark, choose a model, runtime, and target device, then start the benchmark.',
      color: '#0891b2',
      bg: '#ecfeff',
      border: '#a5f3fc',
      targetTab: 'benchmark',
    },
    {
      num: 3,
      title: 'View Results',
      description: 'Monitor progress and view available metrics such as accuracy, latency, memory usage, and model size.',
      color: '#d97706',
      bg: '#fffbeb',
      border: '#fde68a',
      targetTab: 'compare',
    },
    {
      num: 4,
      title: 'Get Recommendations',
      description: 'Use Recommendations to identify suitable models based on available results and edge hardware requirements.',
      color: '#059669',
      bg: '#ecfdf5',
      border: '#a7f3d0',
      targetTab: 'recommendations',
    },
    {
      num: 5,
      title: 'Check Run History',
      description: 'Visit Run History to review previous benchmark sessions and their results.',
      color: '#7c3aed',
      bg: '#f5f3ff',
      border: '#ddd6fe',
      targetTab: 'history',
    },
  ];

  const quickActions = [
    {
      id: 'quick-action-compare',
      title: 'Model Comparison Matrix',
      description: 'Select vision models and runtimes to compare latency, memory delta, accuracy, and storage side-by-side.',
      icon: Layers,
      iconColor: '#2563eb',
      iconBg: '#eff6ff',
      actionText: 'Open Comparison Matrix',
      targetTab: 'compare',
    },
    {
      id: 'quick-action-benchmark',
      title: 'Run Benchmark & Profiling',
      description: 'Configure and trigger empirical Host CPU benchmarks or launch analytical Edge Impulse target profiling.',
      icon: PlayCircle,
      iconColor: '#0891b2',
      iconBg: '#ecfeff',
      actionText: 'Launch Benchmark Runner',
      targetTab: 'benchmark',
    },
    {
      id: 'quick-action-recommendations',
      title: 'Edge Recommendations',
      description: 'Evaluate multi-criteria decision analysis (MCDA) rankings tailored to edge hardware constraints.',
      icon: Award,
      iconColor: '#059669',
      iconBg: '#ecfdf5',
      actionText: 'View Recommendations',
      targetTab: 'recommendations',
    },
    {
      id: 'quick-action-history',
      title: 'Audit Run History',
      description: 'Review historical benchmark sessions, target hardware platforms, execution statuses, and metric logs.',
      icon: History,
      iconColor: '#7c3aed',
      iconBg: '#f5f3ff',
      actionText: 'Explore Run History',
      targetTab: 'history',
    },
  ];

  return (
    <div id="overview-summary-dashboard">
      {/* 1. Summary Metric Cards */}
      <SummaryCards models={models} results={results} totalResults={totalResults} />

      {/* 2. How to Use EdgeBench Card */}
      <div
        id="overview-how-to-use"
        style={{
          background: '#ffffff',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          padding: '1.25rem 1.5rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
          marginBottom: '1.5rem',
        }}
      >
        <div style={{ marginBottom: '1.25rem' }}>
          <h2
            style={{
              fontSize: '1.05rem',
              fontWeight: 800,
              color: '#0f172a',
              letterSpacing: '-0.02em',
              margin: 0,
            }}
          >
            How to Use EdgeBench
          </h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '0.2rem 0 0' }}>
            Follow these simple steps to benchmark and select the best image classification model for your edge deployment.
          </p>
        </div>

        <div className="how-to-use-steps-container">
          {steps.map((step, idx) => (
            <React.Fragment key={step.num}>
              <div
                className="how-to-use-step-item"
                onClick={() => onNavigate && onNavigate(step.targetTab)}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  padding: '1rem',
                  cursor: onNavigate ? 'pointer' : 'default',
                  transition: 'all 150ms ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '0.6rem' }}>
                  <div
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '50%',
                      background: step.bg,
                      border: `1.5px solid ${step.border}`,
                      color: step.color,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: 800,
                      fontSize: '0.85rem',
                      flexShrink: 0,
                    }}
                  >
                    {step.num}
                  </div>
                  <h3
                    style={{
                      fontSize: '0.875rem',
                      fontWeight: 700,
                      color: '#0f172a',
                      margin: 0,
                    }}
                  >
                    {step.title}
                  </h3>
                </div>
                <p
                  style={{
                    fontSize: '0.775rem',
                    color: '#64748b',
                    lineHeight: 1.45,
                    margin: 0,
                    flex: 1,
                  }}
                >
                  {step.description}
                </p>
              </div>

              {idx < steps.length - 1 && (
                <div
                  className="how-to-use-connector"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#cbd5e1',
                    padding: '0 0.15rem',
                    flexShrink: 0,
                  }}
                >
                  <ChevronRight size={18} />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 3. Platform Workspaces & Navigation */}
      <div
        id="overview-quick-actions"
        style={{
          background: '#ffffff',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          padding: '1.25rem 1.5rem',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
          marginBottom: '1.5rem',
        }}
      >
        <div style={{ marginBottom: '1rem' }}>
          <h2
            style={{
              fontSize: '1.05rem',
              fontWeight: 800,
              color: '#0f172a',
              letterSpacing: '-0.02em',
              margin: 0,
            }}
          >
            Platform Workspaces & Navigation
          </h2>
          <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '0.2rem 0 0' }}>
            Direct access to dedicated evaluation, matrix comparison, edge recommendations, and audit history.
          </p>
        </div>

        <div className="overview-workspaces-grid">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <div
                key={action.id}
                id={action.id}
                style={{
                  border: '1px solid #e2e8f0',
                  borderRadius: '6px',
                  padding: '1rem',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  background: '#f8fafc',
                  transition: 'border-color 150ms ease, box-shadow 150ms ease',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
                    <div
                      style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '6px',
                        background: action.iconBg,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                      }}
                    >
                      <Icon size={16} color={action.iconColor} />
                    </div>
                    <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>
                      {action.title}
                    </h3>
                  </div>
                  <p style={{ fontSize: '0.775rem', color: '#64748b', lineHeight: 1.4, margin: '0 0 0.85rem' }}>
                    {action.description}
                  </p>
                </div>

                <button
                  id={`btn-${action.id}`}
                  onClick={() => onNavigate && onNavigate(action.targetTab)}
                  className="btn btn-sm"
                  style={{
                    width: '100%',
                    justifyContent: 'space-between',
                    background: '#ffffff',
                    border: '1px solid #cbd5e1',
                    color: '#2563eb',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    padding: '0.45rem 0.75rem',
                    borderRadius: '5px',
                  }}
                >
                  <span>{action.actionText}</span>
                  <ArrowRight size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
