import React, { useState, useMemo } from 'react';
import { BarChart3, PieChart, Zap, Award, HardDrive, Info, Layers } from 'lucide-react';

// Coordinated color palette for consistent configuration representation across all 3 charts
const CONFIG_PALETTE = [
  { primary: '#2563eb', secondary: '#93c5fd', bg: '#eff6ff', border: '#bfdbfe' }, // Blue
  { primary: '#059669', secondary: '#6ee7b7', bg: '#ecfdf5', border: '#a7f3d0' }, // Emerald
  { primary: '#7c3aed', secondary: '#c4b5fd', bg: '#f5f3ff', border: '#ddd6fe' }, // Purple
  { primary: '#d97706', secondary: '#fcd34d', bg: '#fffbeb', border: '#fde68a' }, // Amber
  { primary: '#0891b2', secondary: '#67e8f9', bg: '#ecfeff', border: '#a5f3fc' }, // Cyan
  { primary: '#dc2626', secondary: '#fca5a5', bg: '#fef2f2', border: '#fecaca' }, // Red
  { primary: '#4f46e5', secondary: '#a5b4fc', bg: '#eef2ff', border: '#c7d2fe' }, // Indigo
  { primary: '#0d9488', secondary: '#5eead4', bg: '#f0fdfa', border: '#99f6e4' }, // Teal
];

export default function VisualCharts({ results = [], sessionInfo = null }) {
  // Chart view mode: 'all' (all 3 charts in responsive grid), 'latency', 'size', or 'accuracy'
  const [activeView, setActiveView] = useState('all');
  // Scale mode for latency: 'auto', 'linear', or 'logarithmic'
  const [latencyScaleMode, setLatencyScaleMode] = useState('auto');
  // Hovered item for interactive tooltip / slice highlight
  const [hoveredKey, setHoveredKey] = useState(null);

  // Normalize and memoize configuration items
  const chartItems = useMemo(() => {
    if (!results || results.length === 0) return [];

    return results.map((r, idx) => {
      const isEdge = r.provenance === 'edge_impulse_estimation' || r.device === 'raspberry-pi-4';
      const colorScheme = CONFIG_PALETTE[idx % CONFIG_PALETTE.length];
      const runId = r.run_id ?? r.id ?? (idx + 1);

      return {
        key: `${r.model_id}-${r.runtime}-${r.device || 'cpu'}-${runId}-${idx}`,
        idx,
        modelId: r.model_id,
        architecture: r.architecture || r.model_id,
        runtime: (r.runtime || '').toUpperCase(),
        device: r.device || 'cpu',
        runId,
        provenance: r.provenance || 'live_execution',
        isEdge,
        color: colorScheme,
        // Latency in ms (null if missing or invalid)
        latencyMs: r.latency?.mean_ms != null && r.latency.mean_ms > 0 ? Number(r.latency.mean_ms.toFixed(2)) : null,
        // Top-1 Accuracy in percent 0..100 (null if missing or invalid)
        top1Accuracy:
          r.accuracy?.top1_accuracy != null && r.accuracy.top1_accuracy > 0
            ? Number((r.accuracy.top1_accuracy * 100).toFixed(2))
            : null,
        // Top-5 Accuracy in percent 0..100 (null if missing or invalid)
        top5Accuracy:
          r.accuracy?.top5_accuracy != null && r.accuracy.top5_accuracy > 0
            ? Number((r.accuracy.top5_accuracy * 100).toFixed(2))
            : null,
        accuracySource: r.accuracy?.accuracy_source || (isEdge ? 'edge_impulse_estimation' : 'live_evaluation'),
        // Model artifact size in MB (null if missing or invalid)
        sizeMb:
          r.model_size?.file_size_mb != null && r.model_size.file_size_mb > 0
            ? Number(r.model_size.file_size_mb.toFixed(2))
            : null,
      };
    });
  }, [results]);

  if (!results || results.length === 0) {
    return (
      <div
        id="visual-charts-empty"
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
          <BarChart3 size={24} />
        </div>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', margin: '0 0 0.4rem' }}>
          No Benchmark Results Selected
        </h3>
        <p style={{ fontSize: '0.85rem', color: '#64748b', maxWidth: '440px', margin: 0 }}>
          Select model configurations in the Saved Results Selection Matrix above and click Compare Selected to
          render interactive performance visualizations.
        </p>
      </div>
    );
  }

  // -------------------------------------------------------------
  // Data analysis for Latency Chart
  // -------------------------------------------------------------
  const validLatencies = chartItems.filter((d) => d.latencyMs !== null);
  const minLatency = validLatencies.length ? Math.min(...validLatencies.map((d) => d.latencyMs)) : 1;
  const maxLatency = validLatencies.length ? Math.max(...validLatencies.map((d) => d.latencyMs)) : 1;
  const latencyRatio = maxLatency / Math.max(0.1, minLatency);

  // Auto-determine if logarithmic scale improves readability (e.g. when ratio > 20x)
  const isLogScale =
    latencyScaleMode === 'logarithmic' || (latencyScaleMode === 'auto' && latencyRatio >= 20);

  // -------------------------------------------------------------
  // Data analysis for Model Artifact Size (Doughnut Chart)
  // -------------------------------------------------------------
  const validSizes = chartItems.filter((d) => d.sizeMb !== null);
  const allSizesValid = chartItems.length > 0 && chartItems.every((d) => d.sizeMb !== null);
  const totalArtifactSizeMb = validSizes.reduce((sum, d) => sum + d.sizeMb, 0);

  // -------------------------------------------------------------
  // Data analysis for Classification Accuracy (Grouped Bar Chart)
  // -------------------------------------------------------------
  const hasAccuracyData = chartItems.some((d) => d.top1Accuracy !== null || d.top5Accuracy !== null);

  return (
    <section
      id="performance-visualizations-section"
      className="saas-card"
      style={{ margin: '0 0 1.5rem', padding: '1.75rem' }}
      aria-label="Performance Visualizations"
    >
      {/* Section Header */}
      <div
        className="saas-card-header"
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          marginBottom: '1.5rem',
          borderBottom: '1px solid #f1f5f9',
          paddingBottom: '1.25rem',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
            <h2
              className="saas-card-title"
              style={{
                fontSize: '1.25rem',
                fontWeight: 800,
                color: '#0f172a',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                margin: 0,
              }}
            >
              <BarChart3 size={22} color="#2563eb" />
              Performance Visualizations
            </h2>
            {sessionInfo && (
              <span
                className="badge font-mono"
                style={{
                  fontSize: '0.725rem',
                  background: '#f1f5f9',
                  color: '#0f172a',
                  border: '1px solid #e2e8f0',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '6px',
                }}
              >
                {sessionInfo.isSingleSession
                  ? `Unified Session: Run #${sessionInfo.runId}`
                  : `Multi-Session Comparison (${sessionInfo.sourceRuns?.length || chartItems.length} Runs)`}
              </span>
            )}
          </div>
          <p
            className="saas-card-subtitle"
            style={{ fontSize: '0.825rem', color: '#64748b', margin: '0.35rem 0 0' }}
          >
            Multi-dimensional empirical benchmarking visualizations across latency, artifact storage footprint,
            and classification accuracy.
          </p>
        </div>

        {/* View Selection Controls */}
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
          {[
            { id: 'all', label: 'All Visualizations (3)', icon: Layers },
            { id: 'latency', label: 'Mean Latency', icon: Zap },
            { id: 'size', label: 'Artifact Size', icon: HardDrive },
            { id: 'accuracy', label: 'Accuracy', icon: Award },
          ].map((btn) => {
            const Icon = btn.icon;
            const isSelected = activeView === btn.id;
            return (
              <button
                key={btn.id}
                id={`btn-view-${btn.id}`}
                onClick={() => setActiveView(btn.id)}
                className={`btn btn-sm ${isSelected ? 'btn-primary' : 'btn-secondary'}`}
                style={{
                  fontSize: '0.775rem',
                  fontWeight: 600,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  padding: '0.4rem 0.75rem',
                  borderRadius: '6px',
                }}
              >
                <Icon size={14} />
                {btn.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Unified Visualizations Grid */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
        }}
      >
        {/* ========================================================================= */}
        {/* VISUALIZATION A: Mean Inference Latency — Bar Chart                       */}
        {/* ========================================================================= */}
        {(activeView === 'all' || activeView === 'latency') && (
          <div
            id="chart-latency-container"
            style={{
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '1.25rem 1.5rem',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
            }}
          >
            {/* Chart Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '0.75rem',
                marginBottom: '1rem',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                  <h3
                    style={{
                      fontSize: '1rem',
                      fontWeight: 700,
                      color: '#0f172a',
                      margin: 0,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.45rem',
                    }}
                  >
                    <Zap size={18} color="#2563eb" />
                    Mean Inference Latency
                  </h3>
                  <span
                    style={{
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      color: '#059669',
                      background: '#ecfdf5',
                      border: '1px solid #a7f3d0',
                      borderRadius: '4px',
                      padding: '0.15rem 0.45rem',
                    }}
                  >
                    Lower is better ↓
                  </span>
                  {isLogScale && (
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        color: '#d97706',
                        background: '#fffbeb',
                        border: '1px solid #fde68a',
                        borderRadius: '4px',
                        padding: '0.15rem 0.45rem',
                      }}
                    >
                      Logarithmic Scale (log10)
                    </span>
                  )}
                </div>
                <p style={{ fontSize: '0.775rem', color: '#64748b', margin: '0.25rem 0 0' }}>
                  Average sample execution time in milliseconds (ms) across evaluated hardware configurations.
                </p>
              </div>

              {/* Latency Scale Toggle Button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ fontSize: '0.725rem', color: '#64748b', fontWeight: 600 }}>Scale:</span>
                <button
                  id="btn-scale-linear"
                  onClick={() => setLatencyScaleMode('linear')}
                  className={`btn btn-sm ${!isLogScale ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ fontSize: '0.7rem', padding: '0.25rem 0.55rem', borderRadius: '4px' }}
                >
                  Linear
                </button>
                <button
                  id="btn-scale-log"
                  onClick={() => setLatencyScaleMode('logarithmic')}
                  className={`btn btn-sm ${isLogScale ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ fontSize: '0.7rem', padding: '0.25rem 0.55rem', borderRadius: '4px' }}
                >
                  Logarithmic
                </button>
              </div>
            </div>

            {/* Latency Bars List */}
            {validLatencies.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
                No measured latency data available for selected configurations.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                {chartItems.map((item) => {
                  const val = item.latencyMs;
                  let pct = 4;
                  if (val !== null) {
                    if (isLogScale) {
                      const logMin = Math.log10(Math.max(0.1, minLatency));
                      const logMax = Math.log10(Math.max(1, maxLatency));
                      const logVal = Math.log10(Math.max(0.1, val));
                      const logRange = logMax - logMin || 1;
                      pct = Math.max(8, Math.min(100, ((logVal - logMin) / logRange) * 85 + 15));
                    } else {
                      pct = Math.max(4, Math.min(100, (val / maxLatency) * 100));
                    }
                  }

                  const isHovered = hoveredKey === item.key;

                  return (
                    <div
                      key={item.key}
                      id={`latency-row-${item.modelId}-${item.runtime}`}
                      onMouseEnter={() => setHoveredKey(item.key)}
                      onMouseLeave={() => setHoveredKey(null)}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'minmax(200px, 300px) 1fr minmax(110px, 140px)',
                        alignItems: 'center',
                        gap: '1rem',
                        fontSize: '0.825rem',
                        background: isHovered ? '#f8fafc' : 'transparent',
                        padding: '0.35rem 0.5rem',
                        borderRadius: '6px',
                        transition: 'background 150ms ease',
                      }}
                      title={`${item.architecture} [${item.runtime}] Run #${item.runId}: ${
                        val !== null ? `${val} ms` : 'Unavailable'
                      } (${item.isEdge ? 'Analytical Estimate' : 'Host CPU Live Execution'})`}
                    >
                      {/* Configuration Identity Label */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', overflow: 'hidden' }}>
                        <div
                          style={{
                            fontWeight: 700,
                            color: '#0f172a',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.45rem',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          <span
                            style={{
                              width: '10px',
                              height: '10px',
                              borderRadius: '50%',
                              backgroundColor: item.color.primary,
                              flexShrink: 0,
                            }}
                          />
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.architecture}</span>
                          <span
                            style={{
                              fontSize: '0.675rem',
                              fontWeight: 700,
                              background: '#f1f5f9',
                              color: '#334155',
                              padding: '0.1rem 0.35rem',
                              borderRadius: '4px',
                            }}
                          >
                            {item.runtime}
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.7rem', color: '#64748b' }}>
                          <span className="font-mono">Run #{item.runId}</span>
                          <span>•</span>
                          {item.isEdge ? (
                            <span style={{ color: '#b45309', fontWeight: 600 }}>RPi 4 (Est.)</span>
                          ) : (
                            <span style={{ color: '#2563eb', fontWeight: 600 }}>Host CPU</span>
                          )}
                        </div>
                      </div>

                      {/* Bar Track & Fill */}
                      <div
                        style={{
                          height: '24px',
                          background: '#f1f5f9',
                          borderRadius: '6px',
                          overflow: 'hidden',
                          border: val !== null ? '1px solid #e2e8f0' : '1px dashed #cbd5e1',
                          position: 'relative',
                        }}
                      >
                        {val !== null ? (
                          <div
                            style={{
                              height: '100%',
                              width: `${pct}%`,
                              background: item.isEdge
                                ? 'linear-gradient(90deg, #d97706 0%, #f59e0b 100%)'
                                : `linear-gradient(90deg, ${item.color.primary} 0%, ${item.color.secondary} 100%)`,
                              borderRadius: '5px',
                              transition: 'width 400ms cubic-bezier(0.4, 0, 0.2, 1)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'flex-end',
                              paddingRight: '0.5rem',
                              boxShadow: isHovered ? '0 0 8px rgba(37, 99, 235, 0.3)' : 'none',
                            }}
                          />
                        ) : (
                          <div
                            style={{
                              height: '100%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '0.7rem',
                              color: '#94a3b8',
                              fontStyle: 'italic',
                            }}
                          >
                            Latency measurement unavailable (N/A)
                          </div>
                        )}
                      </div>

                      {/* Value Display */}
                      <div
                        className="font-mono"
                        style={{
                          textAlign: 'right',
                          fontWeight: 700,
                          fontSize: '0.85rem',
                          color: item.isEdge ? '#b45309' : '#0f172a',
                        }}
                      >
                        {val !== null ? (
                          <>
                            {item.isEdge ? '~' : ''}
                            {val} ms{item.isEdge ? ' (Est.)' : ''}
                          </>
                        ) : (
                          <span style={{ color: '#94a3b8' }}>N/A</span>
                        )}
                      </div>
                    </div>
                  );
                })}

                {/* Axis Scale Footer */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(200px, 300px) 1fr minmax(110px, 140px)',
                    gap: '1rem',
                    fontSize: '0.7rem',
                    color: '#64748b',
                    borderTop: '1px solid #f1f5f9',
                    paddingTop: '0.5rem',
                    marginTop: '0.25rem',
                  }}
                >
                  <div style={{ fontWeight: 600 }}>Axis: Milliseconds (ms)</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 0.25rem' }}>
                    {isLogScale ? (
                      <>
                        <span>1 ms</span>
                        <span>10 ms</span>
                        <span>100 ms</span>
                        <span>1,000 ms</span>
                        <span>10,000+ ms</span>
                      </>
                    ) : (
                      <>
                        <span>0 ms</span>
                        <span>{(maxLatency * 0.25).toFixed(1)} ms</span>
                        <span>{(maxLatency * 0.5).toFixed(1)} ms</span>
                        <span>{(maxLatency * 0.75).toFixed(1)} ms</span>
                        <span>{maxLatency.toFixed(1)} ms</span>
                      </>
                    )}
                  </div>
                  <div style={{ textAlign: 'right', fontWeight: 600 }}>Duration</div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* ROW 2: Responsive Two Columns for Doughnut & Grouped Bar Chart            */}
        {/* ========================================================================= */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: activeView === 'all' ? 'repeat(auto-fit, minmax(380px, 1fr))' : '1fr',
            gap: '1.5rem',
          }}
        >
          {/* ========================================================================= */}
          {/* VISUALIZATION B: Model Artifact Size — Doughnut Chart                     */}
          {/* ========================================================================= */}
          {(activeView === 'all' || activeView === 'size') && (
            <div
              id="chart-size-container"
              style={{
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '1.25rem 1.5rem',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {/* Chart Header */}
              <div style={{ marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <h3
                    style={{
                      fontSize: '1rem',
                      fontWeight: 700,
                      color: '#0f172a',
                      margin: 0,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.45rem',
                    }}
                  >
                    <HardDrive size={18} color="#7c3aed" />
                    Model Artifact Size
                  </h3>
                  <span
                    style={{
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      color: '#7c3aed',
                      background: '#f5f3ff',
                      border: '1px solid #ddd6fe',
                      borderRadius: '4px',
                      padding: '0.15rem 0.45rem',
                    }}
                  >
                    Megabytes (MB)
                  </span>
                </div>
                <p style={{ fontSize: '0.775rem', color: '#64748b', margin: '0.25rem 0 0' }}>
                  Disk footprint breakdown of verified serialized model artifacts across runtimes.
                </p>
              </div>

              {/* Chart Body */}
              {validSizes.length === 0 ? (
                <div style={{ padding: '2.5rem', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
                  No verified model artifact sizes available for selected configurations.
                </div>
              ) : (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    justifyContent: 'space-around',
                    flexWrap: 'wrap',
                    gap: '1.5rem',
                    flex: 1,
                  }}
                >
                  {/* SVG Doughnut */}
                  <div style={{ position: 'relative', width: '220px', height: '220px', flexShrink: 0 }}>
                    <svg
                      width="220"
                      height="220"
                      viewBox="0 0 240 240"
                      role="img"
                      aria-label="Model Artifact Size Doughnut Chart"
                    >
                      {/* Doughnut Hole / Center Circle */}
                      <circle cx="120" cy="120" r="58" fill="#ffffff" />

                      {/* Render Slices */}
                      {validSizes.length === 1 ? (
                        <circle
                          cx="120"
                          cy="120"
                          r="78"
                          fill="none"
                          stroke={validSizes[0].color.primary}
                          strokeWidth="32"
                          strokeDasharray="490"
                          strokeDashoffset="0"
                        />
                      ) : (
                        (() => {
                          let currentAngle = -90; // Start at top 12 o'clock
                          return validSizes.map((item) => {
                            const sliceAngle = (item.sizeMb / totalArtifactSizeMb) * 360;
                            const startAngle = currentAngle;
                            const endAngle = currentAngle + sliceAngle;
                            currentAngle = endAngle;

                            // SVG Arc Path coordinates
                            const R = 94; // Outer radius
                            const r = 62; // Inner radius
                            const cx = 120;
                            const cy = 120;

                            const rad = (a) => (a * Math.PI) / 180;
                            const x1 = cx + R * Math.cos(rad(startAngle));
                            const y1 = cy + R * Math.sin(rad(startAngle));
                            const x2 = cx + R * Math.cos(rad(endAngle));
                            const y2 = cy + R * Math.sin(rad(endAngle));

                            const x3 = cx + r * Math.cos(rad(endAngle));
                            const y3 = cy + r * Math.sin(rad(endAngle));
                            const x4 = cx + r * Math.cos(rad(startAngle));
                            const y4 = cy + r * Math.sin(rad(startAngle));

                            const largeArc = sliceAngle > 180 ? 1 : 0;
                            const d = `M ${x1} ${y1} A ${R} ${R} 0 ${largeArc} 1 ${x2} ${y2} L ${x3} ${y3} A ${r} ${r} 0 ${largeArc} 0 ${x4} ${y4} Z`;

                            const isHovered = hoveredKey === item.key;

                            return (
                              <path
                                key={item.key}
                                d={d}
                                fill={item.color.primary}
                                stroke="#ffffff"
                                strokeWidth="2"
                                style={{
                                  cursor: 'pointer',
                                  transition: 'transform 150ms ease, opacity 150ms ease',
                                  opacity: hoveredKey && !isHovered ? 0.6 : 1,
                                }}
                                onMouseEnter={() => setHoveredKey(item.key)}
                                onMouseLeave={() => setHoveredKey(null)}
                              >
                                <title>{`${item.architecture} [${item.runtime}] Run #${item.runId}: ${item.sizeMb} MB`}</title>
                              </path>
                            );
                          });
                        })()
                      )}

                      {/* Center Content Text */}
                      <text
                        x="120"
                        y="108"
                        textAnchor="middle"
                        style={{ fontSize: '0.675rem', fontWeight: 700, fill: '#64748b', textTransform: 'uppercase' }}
                      >
                        {allSizesValid ? 'Total Size' : 'Verified Total'}
                      </text>
                      <text
                        x="120"
                        y="130"
                        textAnchor="middle"
                        className="font-mono"
                        style={{ fontSize: '1.25rem', fontWeight: 800, fill: '#0f172a' }}
                      >
                        {totalArtifactSizeMb.toFixed(1)} MB
                      </text>
                      <text
                        x="120"
                        y="146"
                        textAnchor="middle"
                        style={{
                          fontSize: '0.65rem',
                          fontWeight: 600,
                          fill: allSizesValid ? '#059669' : '#d97706',
                        }}
                      >
                        {allSizesValid
                          ? `${validSizes.length} Verified`
                          : `${validSizes.length}/${chartItems.length} Available`}
                      </text>
                    </svg>
                  </div>

                  {/* Legend List */}
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.5rem',
                      flex: 1,
                      minWidth: '180px',
                    }}
                  >
                    {chartItems.map((item) => {
                      const isHovered = hoveredKey === item.key;
                      const hasSize = item.sizeMb !== null;
                      const pctOfTotal = hasSize ? ((item.sizeMb / totalArtifactSizeMb) * 100).toFixed(1) : null;

                      return (
                        <div
                          key={item.key}
                          id={`size-legend-${item.modelId}-${item.runtime}`}
                          onMouseEnter={() => setHoveredKey(item.key)}
                          onMouseLeave={() => setHoveredKey(null)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '0.35rem 0.5rem',
                            borderRadius: '6px',
                            background: isHovered ? item.color.bg : '#f8fafc',
                            border: `1px solid ${isHovered ? item.color.border : '#e2e8f0'}`,
                            fontSize: '0.775rem',
                            transition: 'all 150ms ease',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', overflow: 'hidden' }}>
                            <span
                              style={{
                                width: '10px',
                                height: '10px',
                                borderRadius: '50%',
                                backgroundColor: hasSize ? item.color.primary : '#94a3b8',
                                flexShrink: 0,
                              }}
                            />
                            <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              <span style={{ fontWeight: 700, color: '#0f172a' }}>{item.architecture}</span>{' '}
                              <span style={{ fontSize: '0.7rem', color: '#64748b' }}>[{item.runtime}]</span>
                              <span
                                className="font-mono"
                                style={{ fontSize: '0.675rem', color: '#94a3b8', marginLeft: '0.25rem' }}
                              >
                                Run #{item.runId}
                              </span>
                            </div>
                          </div>

                          <div className="font-mono" style={{ fontWeight: 700, textAlign: 'right', flexShrink: 0 }}>
                            {hasSize ? (
                              <span style={{ color: '#0f172a' }}>
                                {item.sizeMb} MB{' '}
                                <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 500 }}>
                                  ({pctOfTotal}%)
                                </span>
                              </span>
                            ) : (
                              <span style={{ color: '#94a3b8', fontSize: '0.725rem' }}>Unavailable</span>
                            )}
                          </div>
                        </div>
                      );
                    })}

                    {!allSizesValid && (
                      <div
                        style={{
                          fontSize: '0.7rem',
                          color: '#d97706',
                          background: '#fffbeb',
                          border: '1px solid #fde68a',
                          padding: '0.35rem 0.55rem',
                          borderRadius: '4px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          marginTop: '0.25rem',
                        }}
                      >
                        <Info size={12} style={{ flexShrink: 0 }} />
                        <span>Total excludes configurations without verified physical artifacts.</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ========================================================================= */}
          {/* VISUALIZATION C: Classification Accuracy — Grouped Bar Chart              */}
          {/* ========================================================================= */}
          {(activeView === 'all' || activeView === 'accuracy') && (
            <div
              id="chart-accuracy-container"
              style={{
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
                padding: '1.25rem 1.5rem',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              {/* Chart Header */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '0.75rem',
                  marginBottom: '1rem',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <h3
                      style={{
                        fontSize: '1rem',
                        fontWeight: 700,
                        color: '#0f172a',
                        margin: 0,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.45rem',
                      }}
                    >
                      <Award size={18} color="#059669" />
                      Classification Accuracy
                    </h3>
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        color: '#059669',
                        background: '#ecfdf5',
                        border: '1px solid #a7f3d0',
                        borderRadius: '4px',
                        padding: '0.15rem 0.45rem',
                      }}
                    >
                      Higher is better ↑
                    </span>
                  </div>
                  <p style={{ fontSize: '0.775rem', color: '#64748b', margin: '0.25rem 0 0' }}>
                    Grouped Top-1 and Top-5 evaluation accuracy (%) on ImageNet-1K.
                  </p>
                </div>

                {/* Accuracy Legend */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span
                      style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '3px',
                        background: '#2563eb',
                        display: 'inline-block',
                      }}
                    />
                    <span style={{ fontWeight: 600, color: '#0f172a' }}>Top-1</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span
                      style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '3px',
                        background: '#0d9488',
                        display: 'inline-block',
                      }}
                    />
                    <span style={{ fontWeight: 600, color: '#0f172a' }}>Top-5</span>
                  </div>
                </div>
              </div>

              {/* Grouped Bars List */}
              {!hasAccuracyData ? (
                <div style={{ padding: '2.5rem', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
                  No validated classification accuracy data available for selected configurations.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1 }}>
                  {chartItems.map((item) => {
                    const isHovered = hoveredKey === item.key;
                    const top1 = item.top1Accuracy;
                    const top5 = item.top5Accuracy;

                    return (
                      <div
                        key={item.key}
                        id={`accuracy-group-${item.modelId}-${item.runtime}`}
                        onMouseEnter={() => setHoveredKey(item.key)}
                        onMouseLeave={() => setHoveredKey(null)}
                        style={{
                          background: isHovered ? '#f8fafc' : 'transparent',
                          padding: '0.5rem 0.65rem',
                          borderRadius: '6px',
                          border: isHovered ? '1px solid #e2e8f0' : '1px solid transparent',
                          transition: 'all 150ms ease',
                        }}
                      >
                        {/* Group Header: Model + Runtime + Provenance Badge */}
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            marginBottom: '0.4rem',
                            fontSize: '0.775rem',
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                            <span
                              style={{
                                width: '8px',
                                height: '8px',
                                borderRadius: '50%',
                                backgroundColor: item.color.primary,
                              }}
                            />
                            <span style={{ fontWeight: 700, color: '#0f172a' }}>{item.architecture}</span>
                            <span
                              style={{
                                fontSize: '0.675rem',
                                fontWeight: 700,
                                background: '#f1f5f9',
                                color: '#334155',
                                padding: '0.05rem 0.35rem',
                                borderRadius: '4px',
                              }}
                            >
                              {item.runtime}
                            </span>
                            <span className="font-mono" style={{ fontSize: '0.7rem', color: '#64748b' }}>
                              Run #{item.runId}
                            </span>
                          </div>

                          {/* Provenance Context Tag */}
                          <div>
                            {item.accuracySource === 'live_evaluation' ? (
                              <span
                                style={{
                                  fontSize: '0.65rem',
                                  fontWeight: 600,
                                  color: '#059669',
                                  background: '#ecfdf5',
                                  border: '1px solid #a7f3d0',
                                  borderRadius: '4px',
                                  padding: '0.1rem 0.35rem',
                                }}
                              >
                                Live ImageNet
                              </span>
                            ) : item.isEdge ? (
                              <span
                                style={{
                                  fontSize: '0.65rem',
                                  fontWeight: 600,
                                  color: '#b45309',
                                  background: '#fffbeb',
                                  border: '1px solid #fde68a',
                                  borderRadius: '4px',
                                  padding: '0.1rem 0.35rem',
                                }}
                                title="Analytical simulation proxy; not physical device evaluation"
                              >
                                Estimated Proxy
                              </span>
                            ) : (
                              <span
                                style={{
                                  fontSize: '0.65rem',
                                  fontWeight: 600,
                                  color: '#2563eb',
                                  background: '#eff6ff',
                                  border: '1px solid #bfdbfe',
                                  borderRadius: '4px',
                                  padding: '0.1rem 0.35rem',
                                }}
                              >
                                Verified Catalog
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Top-1 Bar Row */}
                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '55px 1fr 65px',
                            alignItems: 'center',
                            gap: '0.6rem',
                            fontSize: '0.725rem',
                            marginBottom: '0.3rem',
                          }}
                        >
                          <span style={{ fontWeight: 600, color: '#2563eb' }}>Top-1</span>
                          <div
                            style={{
                              height: '18px',
                              background: '#f1f5f9',
                              borderRadius: '4px',
                              overflow: 'hidden',
                              border: top1 !== null ? '1px solid #e2e8f0' : '1px dashed #cbd5e1',
                              position: 'relative',
                            }}
                          >
                            {top1 !== null ? (
                              <div
                                style={{
                                  height: '100%',
                                  width: `${Math.max(2, Math.min(100, top1))}%`,
                                  background: 'linear-gradient(90deg, #1d4ed8 0%, #3b82f6 100%)',
                                  borderRadius: '3px',
                                  transition: 'width 400ms ease',
                                }}
                              />
                            ) : (
                              <div
                                style={{
                                  height: '100%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  paddingLeft: '0.5rem',
                                  color: '#94a3b8',
                                  fontStyle: 'italic',
                                  fontSize: '0.675rem',
                                }}
                              >
                                Top-1 accuracy unavailable (N/A)
                              </div>
                            )}
                          </div>
                          <div className="font-mono" style={{ textAlign: 'right', fontWeight: 700, color: '#0f172a' }}>
                            {top1 !== null ? `${top1.toFixed(2)}%` : <span style={{ color: '#94a3b8' }}>N/A</span>}
                          </div>
                        </div>

                        {/* Top-5 Bar Row */}
                        <div
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '55px 1fr 65px',
                            alignItems: 'center',
                            gap: '0.6rem',
                            fontSize: '0.725rem',
                          }}
                        >
                          <span style={{ fontWeight: 600, color: '#0d9488' }}>Top-5</span>
                          <div
                            style={{
                              height: '18px',
                              background: '#f1f5f9',
                              borderRadius: '4px',
                              overflow: 'hidden',
                              border: top5 !== null ? '1px solid #e2e8f0' : '1px dashed #cbd5e1',
                              position: 'relative',
                            }}
                          >
                            {top5 !== null ? (
                              <div
                                style={{
                                  height: '100%',
                                  width: `${Math.max(2, Math.min(100, top5))}%`,
                                  background: 'linear-gradient(90deg, #0f766e 0%, #14b8a6 100%)',
                                  borderRadius: '3px',
                                  transition: 'width 400ms ease',
                                }}
                              />
                            ) : (
                              <div
                                style={{
                                  height: '100%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  paddingLeft: '0.5rem',
                                  color: '#94a3b8',
                                  fontStyle: 'italic',
                                  fontSize: '0.675rem',
                                }}
                              >
                                Top-5 accuracy unavailable (N/A)
                              </div>
                            )}
                          </div>
                          <div className="font-mono" style={{ textAlign: 'right', fontWeight: 700, color: '#0f172a' }}>
                            {top5 !== null ? `${top5.toFixed(2)}%` : <span style={{ color: '#94a3b8' }}>N/A</span>}
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {/* Percentage Axis Footer */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '55px 1fr 65px',
                      gap: '0.6rem',
                      fontSize: '0.7rem',
                      color: '#64748b',
                      borderTop: '1px solid #f1f5f9',
                      paddingTop: '0.4rem',
                      marginTop: '0.25rem',
                    }}
                  >
                    <div style={{ fontWeight: 600 }}>Axis: %</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 0.15rem' }}>
                      <span>0%</span>
                      <span>25%</span>
                      <span>50%</span>
                      <span>75%</span>
                      <span>100%</span>
                    </div>
                    <div style={{ textAlign: 'right', fontWeight: 600 }}>ImageNet</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
