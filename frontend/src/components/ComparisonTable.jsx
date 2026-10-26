import React from 'react';
import { Table, Info, AlertTriangle, Database } from 'lucide-react';

export default function ComparisonTable({ results = [], sessionInfo = null }) {
  if (!results || results.length === 0) {
    return (
      <div className="saas-card" style={{ margin: '0 0 1.5rem', padding: '2.5rem', textAlign: 'center' }}>
        <Info size={32} color="#2563eb" style={{ margin: '0 auto 0.75rem' }} />
        <h3 style={{ fontSize: '1.1rem', marginBottom: '0.35rem', color: '#0f172a', fontWeight: 700 }}>
          Select models and runtimes to compare their performance.
        </h3>
        <p style={{ color: '#64748b', fontSize: '0.85rem' }}>
          Choose models and runtimes in the matrix above, then click Compare Selected to view comparative visualizations and tables.
        </p>
      </div>
    );
  }

  const renderBadge = (val, suffix = '', formatter = (v) => v) => {
    if (val === null || val === undefined) {
      return <span className="badge badge-na">N/A</span>;
    }
    return (
      <span className="font-mono" style={{ fontWeight: 600 }}>
        {formatter(val)}{suffix}
      </span>
    );
  };

  const getRuntimeBadge = (runtime) => {
    const rt = (runtime || '').toLowerCase();
    if (rt.includes('onnx')) {
      return <span className="badge font-mono" style={{ background: '#ecfeff', color: '#0891b2', border: '1px solid #a5f3fc' }}>ONNX</span>;
    }
    if (rt.includes('tflite')) {
      return <span className="badge font-mono" style={{ background: '#f0fdf4', color: '#16a34a', border: '1px solid #bbf7d0' }}>TFLite</span>;
    }
    return <span className="badge font-mono" style={{ background: '#eff6ff', color: '#2563eb', border: '1px solid #bfdbfe' }}>PyTorch</span>;
  };

  const getDeviceBadge = (device, isEdgeImpulse) => {
    if (!device) {
      return <span className="badge badge-na">N/A</span>;
    }
    const d = String(device).toLowerCase();
    if (d === 'cpu' || d === 'host_cpu' || d === 'host') {
      return (
        <span
          className="badge font-mono"
          style={{ background: '#f8fafc', color: '#1e293b', border: '1px solid #cbd5e1' }}
        >
          Host CPU
        </span>
      );
    }
    if (d === 'raspberry-pi-4' || d === 'rpi4' || d === 'raspberrypi4') {
      return (
        <span
          className="badge-estimate"
          style={{ fontSize: '0.725rem' }}
          title={isEdgeImpulse ? 'Edge Impulse analytical estimate on Raspberry Pi 4' : 'Raspberry Pi 4 target device'}
        >
          Raspberry Pi 4
        </span>
      );
    }
    return (
      <span className="badge font-mono" style={{ background: '#f8fafc', color: '#334155', border: '1px solid #cbd5e1' }}>
        {device}
      </span>
    );
  };

  const hasEdgeImpulseResults = results.some(
    (r) => r.provenance === 'edge_impulse_estimation' || r.device === 'raspberry-pi-4'
  );

  return (
    <div className="saas-card" style={{ margin: '0 0 1.5rem', padding: '1.75rem' }}>
      <div className="saas-card-header">
        <div>
          <h2 className="saas-card-title">
            <Table size={22} color="#2563eb" />
            Comparative Benchmark Results
          </h2>
          <p className="saas-card-subtitle">
            Side-by-side performance metrics across classification accuracy, execution latency, memory footprint, and storage size.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {sessionInfo && (
            <span className="badge font-mono" style={{ padding: '0.35rem 0.75rem', background: '#f1f5f9', color: '#0f172a' }}>
              {sessionInfo.isSingleSession ? `Session: Run #${sessionInfo.runId}` : `Multi-Session (${sessionInfo.sourceRuns?.length || 0} Runs)`}
            </span>
          )}
          <span className="badge font-mono" style={{ padding: '0.35rem 0.75rem', background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
            {results.length} Selected Configurations
          </span>
        </div>
      </div>

      {/* Session Provenance Banner */}
      {sessionInfo && sessionInfo.isSingleSession && (
        <div
          id="session-provenance-banner"
          style={{
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
            marginBottom: '1rem',
            fontSize: '0.825rem',
            color: '#1e40af',
          }}
        >
          <Database size={16} color="#2563eb" style={{ flexShrink: 0 }} />
          <div>
            <strong>Unified Benchmark Session: Run #{sessionInfo.runId} </strong>
            • Recorded: {sessionInfo.createdAt ? new Date(sessionInfo.createdAt).toLocaleString() : 'Baseline Run'} • All displayed configurations measured under identical session execution settings (Device: CPU, Batch Size: 1).
          </div>
        </div>
      )}

      {/* Edge Impulse Analytical Estimation Notice (Single Remaining Notice) */}
      {hasEdgeImpulseResults && (
        <div
          id="edge-impulse-disclaimer-banner"
          style={{
            background: '#fffbeb',
            border: '1px solid #fde68a',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.65rem',
            marginBottom: '1.25rem',
            fontSize: '0.825rem',
            color: '#92400e',
            lineHeight: '1.5',
          }}
        >
          <AlertTriangle size={18} color="#d97706" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong style={{ color: '#78350f' }}>Edge Impulse Analytical Estimation Notice: </strong>
            Results labeled with <span className="badge-estimate" style={{ fontSize: '0.7rem' }}>RPi 4 (Est.)</span> are analytical estimates calculated by Edge Impulse Studio based on unaccelerated single-core CPU execution. Latency is an analytical estimate, not a physical-device measurement. RAM usage is not modeled on Linux SBC targets and is preserved as N/A. Edge Impulse estimates are distinguishable and must not be ranked or compared directly as equivalent to live host CPU measurements.
          </div>
        </div>
      )}

      {/* Table Container */}
      <div className="saas-table-container">
        <table className="saas-table">
          <thead>
            <tr>
              <th>Model / Architecture</th>
              <th>Runtime</th>
              <th>Device</th>
              <th>Source Run</th>
              <th>Top-1 Accuracy</th>
              <th>Top-5 Accuracy</th>
              <th>Mean Latency</th>
              <th>Throughput</th>
              <th>Peak RAM</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, idx) => {
              const acc = r.accuracy;
              const lat = r.latency;
              const mem = r.memory;
              const isEdgeImpulse = r.provenance === 'edge_impulse_estimation' || r.device === 'raspberry-pi-4';

              return (
                <tr key={`${r.model_id}-${r.runtime}-${r.device || 'cpu'}-${r.run_id || r.id || idx}`}>
                  <td>
                    <div style={{ fontWeight: 600, color: '#0f172a' }}>
                      {r.architecture || r.model_id}
                    </div>
                    <div style={{ fontSize: '0.725rem', color: '#64748b' }}>
                      ID: {r.model_id}
                    </div>
                  </td>
                  <td>
                    {getRuntimeBadge(r.runtime)}
                  </td>
                  <td>
                    {getDeviceBadge(r.device, isEdgeImpulse)}
                  </td>
                  <td>
                    <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a', fontSize: '0.7rem' }}>
                      Run #{r.run_id}
                    </span>
                  </td>
                  <td className="font-mono">
                    {renderBadge(acc?.top1_accuracy, '%', (v) => (v * 100).toFixed(2))}
                  </td>
                  <td className="font-mono">
                    {renderBadge(acc?.top5_accuracy, '%', (v) => (v * 100).toFixed(2))}
                  </td>
                  <td className="font-mono">
                    {isEdgeImpulse ? (
                      <span title="Edge Impulse analytical estimate (single-core unaccelerated execution)">
                        ~{lat?.mean_ms != null ? lat.mean_ms.toFixed(2) : 'N/A'} ms{' '}
                        <span className="badge-estimate" style={{ fontSize: '0.65rem' }}>Est.</span>
                      </span>
                    ) : (
                      renderBadge(lat?.mean_ms, ' ms', (v) => v.toFixed(2))
                    )}
                  </td>
                  <td className="font-mono">
                    {renderBadge(lat?.throughput_fps, ' FPS', (v) => v.toFixed(1))}
                  </td>
                  <td className="font-mono">
                    {isEdgeImpulse ? (
                      <span className="badge badge-na" title="RAM usage unmodeled on Linux SBC (Edge Impulse)">N/A</span>
                    ) : (
                      renderBadge(mem?.peak_ram_mb, ' MiB', (v) => v.toFixed(1))
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
