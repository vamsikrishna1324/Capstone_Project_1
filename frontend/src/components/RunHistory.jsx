import React, { useState } from 'react';
import {
  History,
  Eye,
  CheckCircle2,
  AlertCircle,
  Clock,
  Server,
  ArrowLeft,
  Calendar,
  Layers,
  Database,
  Play,
} from 'lucide-react';
import { fetchBenchmarkRun } from '../api/client';

export default function RunHistory({ results = [], onNavigate }) {
  const [selectedRunDetail, setSelectedRunDetail] = useState(null);
  const [loadingRunId, setLoadingRunId] = useState(null);
  const [inspectError, setInspectError] = useState(null);

  // Group measurements by run_id
  const runMap = {};
  results.forEach((m) => {
    const rid = m.run_id;
    if (!runMap[rid]) {
      runMap[rid] = {
        run_id: rid,
        created_at: m.created_at,
        models: new Set(),
        runtimes: new Set(),
        measurements: [],
        device: m.device,
        provenance: m.provenance,
        status: m.status,
      };
    }
    runMap[rid].models.add(m.model_id);
    runMap[rid].runtimes.add(m.runtime);
    runMap[rid].measurements.push(m);
  });

  const runs = Object.values(runMap).sort((a, b) => b.run_id - a.run_id);

  const handleInspect = async (runId) => {
    setLoadingRunId(runId);
    setInspectError(null);
    try {
      const data = await fetchBenchmarkRun(runId);
      setSelectedRunDetail(data);
    } catch (err) {
      setInspectError(err.message || `Failed to retrieve run #${runId}`);
    } finally {
      setLoadingRunId(null);
    }
  };

  return (
    <div className="saas-card" style={{ margin: '0 0 1.5rem', padding: '1.75rem' }}>
      <div className="saas-card-header">
        <div>
          <h2 className="saas-card-title">
            <History size={22} color="#2563eb" />
            Persisted Benchmark Audit Trail & Session History
          </h2>
          <p className="saas-card-subtitle">
            Complete historical audit trail of recorded benchmark runs, model configurations, and physical/analytical measurements.
          </p>
        </div>
        <span className="badge font-mono" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
          {runs.length} Saved Sessions
        </span>
      </div>

      {inspectError && (
        <div style={{ padding: '0.75rem 1rem', borderRadius: '8px', background: '#fef2f2', border: '1px solid #fecaca', color: '#b91c1c', fontSize: '0.85rem', marginBottom: '1rem' }}>
          {inspectError}
        </div>
      )}

      {selectedRunDetail ? (
        /* Detailed Run Inspection View */
        <div style={{ padding: '1.25rem', border: '1px solid #e2e8f0', borderRadius: '10px', background: '#ffffff' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
            <button
              onClick={() => setSelectedRunDetail(null)}
              className="btn btn-secondary btn-sm"
              style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}
            >
              <ArrowLeft size={14} /> Back to Sessions List
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a' }}>
                Run #{selectedRunDetail.id}
              </span>
              {selectedRunDetail.provenance === 'edge_impulse_estimation' ? (
                <span className="badge-estimate">Analytical Estimate</span>
              ) : selectedRunDetail.provenance === 'live_execution' ? (
                <span className="badge-live">Measured Result</span>
              ) : (
                <span className="badge badge-na">N/A</span>
              )}
              <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1' }}>
                Job: {(selectedRunDetail.status || 'completed').toUpperCase()}
              </span>
              {(() => {
                const mList = selectedRunDetail.measurements || [];
                const anyFailed = mList.some((m) => m.status === 'failed');
                const allFailed = mList.length > 0 && mList.every((m) => m.status === 'failed');
                const evalOutcome =
                  selectedRunDetail.evaluation_status ||
                  (allFailed ? 'failed' : anyFailed ? 'partial' : 'completed');
                if (evalOutcome === 'failed') {
                  return <span className="badge badge-failed">Evaluation: FAILED</span>;
                }
                if (evalOutcome === 'partial') {
                  return (
                    <span
                      className="badge font-mono"
                      style={{ background: '#fffbeb', color: '#b45309', border: '1px solid #fde68a', fontWeight: 600 }}
                    >
                      Evaluation: PARTIAL
                    </span>
                  );
                }
                return <span className="badge badge-completed">Evaluation: COMPLETED</span>;
              })()}
            </div>
          </div>

          <div style={{ marginBottom: '1.25rem' }}>
            <h3 style={{ fontSize: '1.15rem', color: '#0f172a', fontWeight: 700, marginBottom: '0.25rem' }}>
              {selectedRunDetail.run_name || `Benchmark Run #${selectedRunDetail.id}`}
            </h3>
            <p style={{ fontSize: '0.8rem', color: '#64748b' }}>
              Provenance: <strong>{selectedRunDetail.provenance}</strong> • Target Device: <strong>{selectedRunDetail.device}</strong> • Created: {new Date(selectedRunDetail.created_at).toLocaleString()}
            </p>
          </div>

          {(() => {
            const mList = selectedRunDetail.measurements || [];
            const failedMeasurements = mList.filter((m) => m.status === 'failed');
            const hasFailed =
              failedMeasurements.length > 0 ||
              selectedRunDetail.evaluation_status === 'failed' ||
              selectedRunDetail.evaluation_status === 'partial';

            if (hasFailed) {
              const diagError =
                failedMeasurements.map((m) => m.error_message).find(Boolean) ||
                selectedRunDetail.error_message;

              return (
                <div
                  id="session-evaluation-failure-banner"
                  style={{
                    marginBottom: '1.25rem',
                    padding: '0.85rem 1rem',
                    borderRadius: '8px',
                    background: '#fffbeb',
                    border: '1px solid #fde68a',
                    color: '#92400e',
                    fontSize: '0.85rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                    <AlertCircle size={18} color="#d97706" />
                    <strong>Evaluation Outcome Notice:</strong>
                  </div>
                  <p style={{ margin: '0 0 0.35rem', lineHeight: '1.45', color: '#78350f' }}>
                    One or more model measurements in this session failed benchmark evaluation (e.g. accuracy evaluation dataset was not provided).
                    Any empirical latency, throughput, or memory footprint measurements recorded prior to failure are preserved in the table below,
                    but accuracy evaluation is marked unavailable (N/A).
                  </p>
                  {diagError && (
                    <div
                      style={{
                        fontFamily: 'monospace',
                        fontSize: '0.78rem',
                        background: '#fef3c7',
                        padding: '0.4rem 0.6rem',
                        borderRadius: '4px',
                        border: '1px solid #fcd34d',
                        color: '#78350f',
                        wordBreak: 'break-all',
                      }}
                    >
                      Diagnostic: {diagError}
                    </div>
                  )}
                </div>
              );
            }
            return null;
          })()}

          <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.5rem', color: '#0f172a' }}>
            Associated Measurements ({selectedRunDetail.measurements?.length || 0})
          </h4>
          <div className="saas-table-container">
            <table className="saas-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Runtime</th>
                  <th>Top-1 Acc</th>
                  <th>Mean Latency</th>
                  <th>Throughput</th>
                  <th>Peak RAM</th>
                  <th>Model Size</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {selectedRunDetail.measurements?.map((m) => (
                  <tr key={m.id}>
                    <td>
                      <strong>{m.architecture || m.model_id}</strong>
                      {m.status === 'failed' && m.error_message && (
                        <div style={{ fontSize: '0.72rem', color: '#b91c1c', marginTop: '0.15rem' }}>
                          {m.error_message}
                        </div>
                      )}
                    </td>
                    <td>
                      <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a' }}>
                        {m.runtime.toUpperCase()}
                      </span>
                    </td>
                    <td className="font-mono">
                      {m.accuracy?.top1_accuracy != null ? `${(m.accuracy.top1_accuracy * 100).toFixed(2)}%` : <span className="badge badge-na">N/A</span>}
                    </td>
                    <td className="font-mono">
                      {m.latency?.mean_ms != null ? `${m.latency.mean_ms.toFixed(2)} ms` : <span className="badge badge-na">N/A</span>}
                    </td>
                    <td className="font-mono">
                      {m.latency?.throughput_fps != null ? `${m.latency.throughput_fps.toFixed(1)} fps` : <span className="badge badge-na">N/A</span>}
                    </td>
                    <td className="font-mono">
                      {m.memory?.peak_ram_mb != null ? `${m.memory.peak_ram_mb.toFixed(1)} MiB` : <span className="badge badge-na">N/A</span>}
                    </td>
                    <td className="font-mono">
                      {m.model_size?.file_size_mb != null ? `${m.model_size.file_size_mb.toFixed(2)} MB` : <span className="badge badge-na">N/A</span>}
                    </td>
                    <td>
                      <span className={`badge ${m.status === 'completed' ? 'badge-completed' : 'badge-failed'}`}>
                        {m.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : runs.length === 0 ? (
        <div
          id="history-empty-state"
          style={{
            textAlign: 'center',
            padding: '3rem 1.5rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#ffffff',
            borderRadius: '10px',
            border: '1px solid #e2e8f0',
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
            <History size={24} />
          </div>
          <h3 style={{ fontSize: '1.15rem', color: '#0f172a', marginBottom: '0.4rem', fontWeight: 700 }}>
            No Benchmark Sessions Recorded Yet
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.85rem', maxWidth: '520px', lineHeight: '1.5', margin: '0 auto 1.25rem' }}>
            Execute a benchmark run from the <strong>Run Benchmark</strong> tab to start recording model profiling sessions and metric audit history in MySQL.
          </p>
          {onNavigate && (
            <button
              id="btn-history-run-benchmark"
              onClick={() => onNavigate('benchmark')}
              className="btn btn-primary btn-sm"
              style={{ fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
            >
              <Play size={14} fill="currentColor" />
              Go to Run Benchmark
            </button>
          )}
        </div>
      ) : (
        /* Sessions Table */
        <div className="saas-table-container">
          <table className="saas-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Models Included</th>
                <th>Runtimes</th>
                <th>Target Device</th>
                <th>Provenance</th>
                <th>Measurements</th>
                <th>Status</th>
                <th>Recorded Timestamp</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a', fontWeight: 700 }}>
                      #{r.run_id}
                    </span>
                  </td>
                  <td>
                    <span style={{ fontWeight: 600 }}>{Array.from(r.models).join(', ')}</span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                      {Array.from(r.runtimes).map((rt) => (
                        <span key={rt} className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a', fontSize: '0.7rem' }}>
                          {rt.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <span className="font-mono">{r.device || 'cpu'}</span>
                  </td>
                  <td>
                    {r.provenance === 'edge_impulse_estimation' ? (
                      <span className="badge-estimate">Analytical Estimate</span>
                    ) : r.provenance === 'live_execution' ? (
                      <span className="badge-live">Measured Result</span>
                    ) : (
                      <span className="badge badge-na">N/A</span>
                    )}
                  </td>
                  <td>
                    <span className="font-mono" style={{ fontWeight: 600 }}>{r.measurements.length}</span>
                  </td>
                  <td>
                    {(() => {
                      const anyFailed = r.measurements.some((m) => m.status === 'failed');
                      const allFailed = r.measurements.length > 0 && r.measurements.every((m) => m.status === 'failed');
                      if (allFailed) {
                        return <span className="badge badge-failed">Failed</span>;
                      }
                      if (anyFailed) {
                        return (
                          <span
                            className="badge font-mono"
                            style={{ background: '#fffbeb', color: '#b45309', border: '1px solid #fde68a', fontSize: '0.75rem' }}
                          >
                            Partial
                          </span>
                        );
                      }
                      return <span className="badge badge-completed">Completed</span>;
                    })()}
                  </td>
                  <td style={{ fontSize: '0.8rem', color: '#64748b' }}>
                    {r.created_at ? new Date(r.created_at).toLocaleString() : 'N/A'}
                  </td>
                  <td>
                    <button
                      id={`btn-inspect-run-${r.run_id}`}
                      onClick={() => handleInspect(r.run_id)}
                      disabled={loadingRunId === r.run_id}
                      className="btn btn-secondary btn-sm"
                      style={{ padding: '0.3rem 0.65rem', fontSize: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                    >
                      <Eye size={13} />
                      {loadingRunId === r.run_id ? 'Loading...' : 'Inspect'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
