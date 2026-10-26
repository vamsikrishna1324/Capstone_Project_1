import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Sidebar from './components/Sidebar';
import DashboardHeader from './components/DashboardHeader';
import OverviewDashboard from './components/OverviewDashboard';
import SummaryCards from './components/SummaryCards';
import ModelSelector from './components/ModelSelector';
import BenchmarkRunner from './components/BenchmarkRunner';
import ComparisonTable from './components/ComparisonTable';
import VisualCharts from './components/VisualCharts';
import Recommendations from './components/Recommendations';
import RunHistory from './components/RunHistory';
import { fetchHealth, fetchModels, fetchResults, fetchEdgeImpulseTargets } from './api/client';
import { RefreshCw, AlertCircle, ExternalLink, ShieldCheck, BarChart2 } from 'lucide-react';

function areConfigsEqual(a = {}, b = {}) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  const keysA = Object.keys(a).filter((k) => a[k] && a[k].length > 0);
  const keysB = Object.keys(b).filter((k) => b[k] && b[k].length > 0);
  if (keysA.length !== keysB.length) return false;
  for (const k of keysA) {
    const listA = [...(a[k] || [])].sort();
    const listB = [...(b[k] || [])].sort();
    if (listA.length !== listB.length) return false;
    for (let i = 0; i < listA.length; i++) {
      if (listA[i] !== listB[i]) return false;
    }
  }
  return true;
}

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [health, setHealth] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => {
    try {
      return localStorage.getItem('edgebench_sidebar_collapsed') === 'true';
    } catch (_) {
      return false;
    }
  });

  const handleToggleSidebarCollapse = () => {
    setIsSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('edgebench_sidebar_collapsed', String(next));
      } catch (_) {}
      return next;
    });
  };

  const [models, setModels] = useState([]);
  const [results, setResults] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [targets, setTargets] = useState([]);

  // User's working selection in the Matrix
  const [selectedConfigs, setSelectedConfigs] = useState({});

  // Snapshot of configs locked when "Compare Selected" was clicked
  const [activeComparisonConfigs, setActiveComparisonConfigs] = useState(null);
  const [hasRunComparison, setHasRunComparison] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // Load initial catalog, health status, measurements, and supported targets
  const loadData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const [healthData, modelsData, resultsData, targetsData] = await Promise.all([
        fetchHealth().catch((err) => ({ status: 'error', api: 'unreachable', database: { connected: false } })),
        fetchModels(),
        fetchResults({ limit: 100 }),
        fetchEdgeImpulseTargets().catch(() => ({ targets: [], total_targets: 0 })),
      ]);

      setHealth(healthData);
      setModels(modelsData.models || []);
      setResults(resultsData.items || []);
      setTotalResults(resultsData.total || 0);
      setTargets(targetsData.targets || []);
    } catch (err) {
      setError(err.message || 'Failed to initialize benchmark dashboard.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Count active working selection
  const selectedConfigCount = Object.values(selectedConfigs).reduce(
    (acc, list) => acc + (list ? list.length : 0),
    0
  );

  // Detect whether current working selection has diverged from active comparison
  const isSelectionStale =
    hasRunComparison &&
    activeComparisonConfigs !== null &&
    !areConfigsEqual(selectedConfigs, activeComparisonConfigs);

  // Trigger comparison for the current selection
  const handleInitiateComparison = () => {
    if (selectedConfigCount === 0) return;
    setActiveComparisonConfigs(JSON.parse(JSON.stringify(selectedConfigs)));
    setHasRunComparison(true);
  };

  // Resolve comparison results: prefer single unified session, fallback to explicit multi-session
  const { comparisonResults, sessionInfo } = useMemo(() => {
    if (!hasRunComparison || !activeComparisonConfigs) {
      return { comparisonResults: [], sessionInfo: null };
    }

    const targetPairs = [];
    Object.entries(activeComparisonConfigs).forEach(([mId, runtimes]) => {
      if (Array.isArray(runtimes)) {
        runtimes.forEach((rtToken) => {
          if (rtToken && rtToken.includes(':')) {
            const [rt, dev] = rtToken.split(':');
            targetPairs.push({ modelId: mId, runtime: rt, device: dev });
          } else {
            targetPairs.push({ modelId: mId, runtime: rtToken, device: null });
          }
        });
      }
    });

    if (targetPairs.length === 0) {
      return { comparisonResults: [], sessionInfo: null };
    }

    // Filter only eligible, completed measurements under established rules
    const eligibleResults = results.filter(
      (m) =>
        (m.status === 'completed' || (!m.status && (m.latency?.mean_ms != null || m.latency_mean_ms != null))) &&
        m.status !== 'failed' &&
        m.status !== 'pending' &&
        m.status !== 'running'
    );

    const runsMap = new Map();
    eligibleResults.forEach((m) => {
      if (!runsMap.has(m.run_id)) {
        runsMap.set(m.run_id, {
          runId: m.run_id,
          createdAt: m.created_at,
          device: m.device,
          provenance: m.provenance,
          measurements: new Map(),
        });
      }
      const run = runsMap.get(m.run_id);
      const key = `${m.model_id}:${m.runtime}`;
      if (!run.measurements.has(key)) {
        run.measurements.set(key, m);
      }
    });

    // Check candidate runs in descending order of createdAt/runId
    const candidateRuns = Array.from(runsMap.values()).sort((a, b) => {
      const tA = new Date(a.createdAt || 0).getTime();
      const tB = new Date(b.createdAt || 0).getTime();
      if (tB !== tA) return tB - tA;
      return (b.runId || 0) - (a.runId || 0);
    });

    // Prefer standardized Host CPU sessions for model matrix comparison
    const hostCandidateRuns = candidateRuns.filter(
      (r) => r.provenance !== 'edge_impulse_estimation' && (r.device == null || r.device === 'cpu')
    );
    const searchRuns = hostCandidateRuns.length > 0 ? hostCandidateRuns : candidateRuns;

    let singleSessionRun = null;
    for (const run of searchRuns) {
      const hasAll = targetPairs.every((p) => {
        const m = run.measurements.get(`${p.modelId}:${p.runtime}`);
        if (!m) return false;
        if (p.device && (m.device || 'cpu') !== p.device) return false;
        return true;
      });
      if (hasAll) {
        singleSessionRun = run;
        break;
      }
    }

    if (singleSessionRun) {
      const matched = targetPairs.map((p) => singleSessionRun.measurements.get(`${p.modelId}:${p.runtime}`));
      return {
        comparisonResults: matched,
        sessionInfo: {
          isSingleSession: true,
          runId: singleSessionRun.runId,
          createdAt: singleSessionRun.createdAt,
          sourceRuns: [singleSessionRun.runId],
        },
      };
    }

    // Fallback: Sourced across historical runs with explicit source runs
    const matched = [];
    const sourceRunIds = new Set();
    targetPairs.forEach((p) => {
      let candidates = eligibleResults.filter(
        (m) => m.model_id === p.modelId && m.runtime === p.runtime
      );
      if (p.device) {
        const devCandidates = candidates.filter((m) => (m.device || 'cpu') === p.device);
        if (devCandidates.length > 0) candidates = devCandidates;
      } else {
        // Prefer Host CPU live execution results so Edge Impulse estimates do not displace host baselines
        const hostCandidates = candidates.filter(
          (m) => m.provenance !== 'edge_impulse_estimation' && (m.device == null || m.device === 'cpu')
        );
        if (hostCandidates.length > 0) candidates = hostCandidates;
      }

      if (candidates.length > 0) {
        const sorted = [...candidates].sort((a, b) => {
          const tA = new Date(a.created_at || 0).getTime();
          const tB = new Date(b.created_at || 0).getTime();
          if (tB !== tA) return tB - tA;
          return (b.id || 0) - (a.id || 0);
        });
        const chosen = sorted[0];
        matched.push(chosen);
        sourceRunIds.add(chosen.run_id);
      }
    });

    return {
      comparisonResults: matched,
      sessionInfo: {
        isSingleSession: false,
        runId: null,
        createdAt: null,
        sourceRuns: Array.from(sourceRunIds).sort((a, b) => b - a),
      },
    };
  }, [hasRunComparison, activeComparisonConfigs, results]);

  const selectedModelIds = Object.keys(selectedConfigs).filter(
    (k) => selectedConfigs[k] && selectedConfigs[k].length > 0
  );

  return (
    <div className="app-shell">
      {/* Dark Vertical Navigation Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        health={health}
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={handleToggleSidebarCollapse}
      />

      <div className="app-main-canvas">
        {/* Light Theme Dashboard Topbar */}
        <DashboardHeader
          activeTab={activeTab}
          onRefresh={() => loadData(true)}
          refreshing={refreshing}
          onToggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        />

        {/* Error Banner */}
        {error && (
          <div
            id="dashboard-error-banner"
            style={{
              margin: '1.25rem 1.75rem 0',
              padding: '1rem 1.25rem',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#b91c1c',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
            }}
          >
            <AlertCircle size={20} style={{ flexShrink: 0 }} />
            <div style={{ flex: 1, fontSize: '0.875rem' }}>
              <strong>Dashboard Connection Warning: </strong>
              {error}
            </div>
            <button onClick={() => loadData()} className="btn btn-sm btn-secondary">
              Retry
            </button>
          </div>
        )}

        {/* Dashboard View Switching */}
        <main className="app-content-body">
        {activeTab === 'dashboard' && (
          <OverviewDashboard
            models={models}
            results={results}
            totalResults={totalResults}
            onNavigate={setActiveTab}
          />
        )}

        {activeTab === 'compare' && (
          <>
            <ModelSelector
              models={models}
              savedResults={results}
              selectedConfigs={selectedConfigs}
              onChangeConfigs={setSelectedConfigs}
              onInitiateComparison={handleInitiateComparison}
              hasActiveComparison={hasRunComparison}
              isSelectionStale={isSelectionStale}
              onNavigate={setActiveTab}
            />

            {hasRunComparison && comparisonResults.length > 0 ? (
              <>
                <ComparisonTable results={comparisonResults} sessionInfo={sessionInfo} />
                <VisualCharts results={comparisonResults} sessionInfo={sessionInfo} />
              </>
            ) : selectedConfigCount > 0 ? (
              <div
                id="compare-empty-state"
                className="saas-card"
                style={{
                  margin: '0 0 1.5rem',
                  padding: '3rem 1.5rem',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: '#eff6ff',
                    border: '1px solid #bfdbfe',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1rem',
                    color: '#2563eb',
                  }}
                >
                  <BarChart2 size={24} />
                </div>
                <h3 style={{ fontSize: '1.15rem', color: '#0f172a', marginBottom: '0.4rem', fontWeight: 700 }}>
                  Ready to Compare {selectedConfigCount} Selected Configuration{selectedConfigCount > 1 ? 's' : ''}
                </h3>
                <p style={{ color: '#64748b', fontSize: '0.85rem', maxWidth: '520px', lineHeight: '1.5', margin: '0 auto 1.25rem' }}>
                  Click <strong>Compare Selected</strong> above or below to generate performance visualizations and comparative measurement tables.
                </p>
                <button
                  id="btn-compare-selected-empty"
                  onClick={handleInitiateComparison}
                  className="btn btn-primary btn-sm"
                  style={{ fontWeight: 600, padding: '0.45rem 1.15rem' }}
                >
                  Compare Selected ({selectedConfigCount} Configs)
                </button>
              </div>
            ) : null}
          </>
        )}

        {activeTab === 'benchmark' && (
          <BenchmarkRunner
            models={models}
            targets={targets}
            savedResults={results}
            onBenchmarkSuccess={(newRun) => {
              loadData(true);
            }}
            onNavigate={setActiveTab}
          />
        )}

        {activeTab === 'recommendations' && (
          <Recommendations
            selectedModelIds={selectedModelIds}
            savedResults={results}
            onNavigate={setActiveTab}
          />
        )}

        {activeTab === 'history' && (
          <RunHistory
            results={results}
            onNavigate={setActiveTab}
          />
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid #e2e8f0',
          padding: '1.25rem 1.75rem',
          marginTop: 'auto',
          fontSize: '0.8rem',
          color: '#64748b',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          background: '#ffffff',
        }}
      >
        <div>
          B.Tech Capstone Project — Automated Image Model Benchmarking for Edge Deployment.
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#64748b', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
          >
            FastAPI Swagger Docs <ExternalLink size={12} />
          </a>
          <span style={{ color: '#cbd5e1' }}>•</span>
          <span>MySQL 8.0 Persistence</span>
        </div>
      </footer>
      </div>
    </div>
  );
}
