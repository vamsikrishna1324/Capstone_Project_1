import React, { useState } from 'react';
import {
  Award,
  Sliders,
  ShieldCheck,
  AlertTriangle,
  Zap,
  Layers,
  HardDrive,
  Database,
  Cpu,
  Info,
  Play,
  Check,
  XCircle,
  RefreshCw,
} from 'lucide-react';
import { fetchRecommendation } from '../api/client';

const CANDIDATE_COLORS = [
  { primary: '#2563eb', light: '#eff6ff', border: '#bfdbfe', badge: '#1d4ed8', label: '#1 • Best Match' },
  { primary: '#4f46e5', light: '#eef2ff', border: '#c7d2fe', badge: '#3730a3', label: '#2 • Alternative' },
  { primary: '#0d9488', light: '#f0fdfa', border: '#99f6e4', badge: '#115e59', label: '#3 • Alternative' },
];

/**
 * Compact visual comparison chart across recommended candidates.
 */
function RecommendationComparisonVisual({ candidates }) {
  if (!candidates || candidates.length === 0) return null;

  const metricConfigs = [
    {
      key: 'latency',
      label: 'Mean Latency',
      unit: 'ms',
      direction: 'Lower is better ↓',
      directionType: 'lower',
      format: (val) => `${val.toFixed(2)} ms`,
    },
    {
      key: 'memory',
      label: 'RAM Footprint',
      unit: 'MiB',
      direction: 'Lower is better ↓',
      directionType: 'lower',
      format: (val) => `${val.toFixed(1)} MiB`,
    },
    {
      key: 'model_size',
      label: 'Artifact Size',
      unit: 'MB',
      direction: 'Lower is better ↓',
      directionType: 'lower',
      format: (val) => `${val.toFixed(2)} MB`,
    },
    {
      key: 'accuracy',
      label: 'Top-1 Accuracy',
      unit: '%',
      direction: 'Higher is better ↑',
      directionType: 'higher',
      format: (val) => `${(val * 100).toFixed(1)}%`,
    },
  ];

  return (
    <div
      id="recommendations-visual-comparison"
      style={{
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '1.75rem',
        boxShadow: '0 2px 6px rgba(0, 0, 0, 0.03)',
      }}
    >
      <div style={{ marginBottom: '1.25rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0f172a', margin: '0 0 0.25rem' }}>
          Visual Comparison of Recommended Candidates
        </h3>
        <p style={{ fontSize: '0.825rem', color: '#64748b', margin: 0 }}>
          Side-by-side normalized performance scaling across verified hardware metrics. Unmeasured dimensions are explicitly designated as N/A.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
        {metricConfigs.map((cfg) => {
          const validValues = candidates
            .map((c) => c.raw_metrics?.[cfg.key])
            .filter((v) => v !== null && v !== undefined && typeof v === 'number');

          const maxVal = validValues.length > 0 ? Math.max(...validValues) : 0;

          return (
            <div
              key={cfg.key}
              style={{
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                padding: '1rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1e293b' }}>
                  {cfg.label}
                </span>
                <span
                  style={{
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    padding: '0.15rem 0.45rem',
                    borderRadius: '4px',
                    background: cfg.directionType === 'lower' ? '#f0fdf4' : '#eff6ff',
                    color: cfg.directionType === 'lower' ? '#166534' : '#1d4ed8',
                    border: `1px solid ${cfg.directionType === 'lower' ? '#bbf7d0' : '#bfdbfe'}`,
                  }}
                >
                  {cfg.direction}
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                {candidates.map((cand, idx) => {
                  const val = cand.raw_metrics?.[cfg.key];
                  const colorScheme = CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length];
                  const hasVal = val !== null && val !== undefined && typeof val === 'number';

                  let pct = 0;
                  if (hasVal) {
                    if (cfg.directionType === 'higher') {
                      pct = Math.max(10, Math.min(100, Math.round((val / (maxVal || 1)) * 100)));
                    } else {
                      pct = Math.max(10, Math.min(100, Math.round((val / (maxVal || 1)) * 100)));
                    }
                  }

                  return (
                    <div key={`${cand.model_id}-${cand.runtime}`}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.2rem' }}>
                        <span style={{ fontWeight: 600, color: '#334155' }}>
                          #{idx + 1} {cand.display_name} [{cand.runtime?.toUpperCase()}]
                        </span>
                        <span className="font-mono" style={{ fontWeight: 700, color: hasVal ? '#0f172a' : '#94a3b8' }}>
                          {hasVal ? cfg.format(val) : 'N/A (Not measured)'}
                        </span>
                      </div>

                      {hasVal ? (
                        <div
                          style={{
                            background: '#e2e8f0',
                            borderRadius: '4px',
                            height: '8px',
                            overflow: 'hidden',
                          }}
                        >
                          <div
                            style={{
                              width: `${pct}%`,
                              height: '100%',
                              background: colorScheme.primary,
                              borderRadius: '4px',
                              transition: 'width 300ms ease',
                            }}
                          />
                        </div>
                      ) : (
                        <div
                          style={{
                            height: '8px',
                            background: '#f1f5f9',
                            border: '1px dashed #cbd5e1',
                            borderRadius: '4px',
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function getCandidateRationale(cand, idx, criteriaWeights) {
  if (cand.rationale) return cand.rationale;

  const sortedWeights = Object.entries(criteriaWeights || {})
    .filter(([_, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k.replace('_', ' ')} (${Math.round(v * 100)}%)`);
  const weightsSummary = sortedWeights.join(', ') || 'balanced criteria';

  const strengths = [];
  if (cand.raw_metrics?.accuracy != null) strengths.push(`${(cand.raw_metrics.accuracy * 100).toFixed(1)}% Top-1 accuracy`);
  if (cand.raw_metrics?.latency != null) strengths.push(`${cand.raw_metrics.latency.toFixed(2)}ms latency`);
  if (cand.raw_metrics?.memory != null) strengths.push(`${cand.raw_metrics.memory.toFixed(1)}MB RAM`);
  if (cand.raw_metrics?.model_size != null) strengths.push(`${cand.raw_metrics.model_size.toFixed(2)}MB size`);
  const strengthsStr = strengths.join(', ') || 'standard edge profile';

  if (idx === 0) {
    return `Top-ranked match scoring ${(cand.composite_score || 0).toFixed(1)}/100 aligned with requested priorities (${weightsSummary}). Delivers ${strengthsStr}.`;
  }
  return `Rank #${cand.rank || idx + 1} eligible option scoring ${(cand.composite_score || 0).toFixed(1)}/100 based on ${weightsSummary}. Features ${strengthsStr}.`;
}

function getCandidateTradeOffs(cand, idx, recData) {
  if (cand.trade_offs) return cand.trade_offs;
  if (idx === 0) return recData?.trade_off_analysis || 'Optimal edge configuration satisfying operational criteria.';
  return 'Viable alternative with balanced edge trade-offs.';
}

export default function Recommendations({
  selectedModelIds = [],
  savedResults = [],
  onNavigate,
}) {
  const [preset, setPreset] = useState('balanced');
  const [weights, setWeights] = useState({
    accuracy: 0.35,
    latency: 0.30,
    memory: 0.20,
    model_size: 0.15,
  });

  const [constraints, setConstraints] = useState({
    maxLatency: '',
    maxMemory: '',
    maxSize: '',
    minAccuracy: '',
  });

  const [loading, setLoading] = useState(false);
  const [recData, setRecData] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [isStale, setIsStale] = useState(false);

  const presets = [
    { id: 'balanced', label: 'Balanced', desc: 'Standard multi-criteria trade-offs' },
    { id: 'speed_optimized', label: 'Speed Optimized', desc: '75% Latency weight prioritizes real-time throughput' },
    { id: 'edge_constrained', label: 'Edge Constrained', desc: '40% Size + 30% Memory for strict embedded RAM/ROM' },
    { id: 'accuracy_first', label: 'Accuracy First', desc: '75% Accuracy weight for critical vision tasks' },
    { id: 'custom', label: 'Custom Weights', desc: 'Fine-tune exact criteria percentages' },
  ];

  const handlePresetSelect = (id) => {
    setPreset(id);
    setValidationError(null);
    if (hasGenerated) setIsStale(true);

    if (id === 'balanced') {
      setWeights({ accuracy: 0.35, latency: 0.30, memory: 0.20, model_size: 0.15 });
    } else if (id === 'speed_optimized') {
      setWeights({ accuracy: 0.10, latency: 0.75, memory: 0.075, model_size: 0.075 });
    } else if (id === 'edge_constrained') {
      setWeights({ accuracy: 0.10, latency: 0.20, memory: 0.30, model_size: 0.40 });
    } else if (id === 'accuracy_first') {
      setWeights({ accuracy: 0.75, latency: 0.10, memory: 0.075, model_size: 0.075 });
    }
  };

  const handleWeightChange = (key, val) => {
    setWeights((prev) => ({ ...prev, [key]: val }));
    setValidationError(null);
    if (hasGenerated) setIsStale(true);
  };

  const handleConstraintChange = (key, val) => {
    setConstraints((prev) => ({ ...prev, [key]: val }));
    setValidationError(null);
    if (hasGenerated) setIsStale(true);
  };

  const handleGenerate = async () => {
    setValidationError(null);
    setErrorMsg(null);

    // Validation: Custom weights cannot all be zero
    if (preset === 'custom') {
      const sumWeights =
        (weights.accuracy || 0) +
        (weights.latency || 0) +
        (weights.memory || 0) +
        (weights.model_size || 0);

      if (sumWeights <= 0) {
        setValidationError('Set at least one criterion weight above zero.');
        return;
      }
    }

    // Validation: Constraints must be valid positive numbers when supplied
    if (constraints.maxLatency !== '') {
      const num = parseFloat(constraints.maxLatency);
      if (isNaN(num) || num <= 0) {
        setValidationError('Maximum latency must be a positive number greater than 0.');
        return;
      }
    }
    if (constraints.maxMemory !== '') {
      const num = parseFloat(constraints.maxMemory);
      if (isNaN(num) || num <= 0) {
        setValidationError('Maximum memory must be a positive number greater than 0.');
        return;
      }
    }
    if (constraints.maxSize !== '') {
      const num = parseFloat(constraints.maxSize);
      if (isNaN(num) || num <= 0) {
        setValidationError('Maximum artifact size must be a positive number greater than 0.');
        return;
      }
    }
    if (constraints.minAccuracy !== '') {
      const num = parseFloat(constraints.minAccuracy);
      if (isNaN(num) || num < 0 || num > 100) {
        setValidationError('Minimum accuracy must be between 0% and 100%.');
        return;
      }
    }

    setLoading(true);

    const payload = {
      model_ids: selectedModelIds && selectedModelIds.length > 0 ? selectedModelIds : undefined,
      priority_preset: preset !== 'custom' ? preset : undefined,
      weights: preset === 'custom' ? weights : undefined,
      constraints: {
        max_latency_ms: constraints.maxLatency ? parseFloat(constraints.maxLatency) : undefined,
        max_memory_mb: constraints.maxMemory ? parseFloat(constraints.maxMemory) : undefined,
        max_model_size_mb: constraints.maxSize ? parseFloat(constraints.maxSize) : undefined,
        min_accuracy: constraints.minAccuracy ? parseFloat(constraints.minAccuracy) / 100 : undefined,
      },
    };

    try {
      const data = await fetchRecommendation(payload);
      setRecData(data);
      setHasGenerated(true);
      setIsStale(false);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to generate recommendation.');
    } finally {
      setLoading(false);
    }
  };

  // Extract top recommendations (up to 3 eligible options)
  const topRecommendations =
    recData?.top_recommendations && recData.top_recommendations.length > 0
      ? recData.top_recommendations
      : (recData?.rankings || []).filter((r) => r.eligible).slice(0, 3);

  return (
    <div className="saas-card" style={{ margin: '0 0 1.5rem', padding: '1.75rem' }}>
      <div className="saas-card-header">
        <div>
          <h2 className="saas-card-title">
            <Award size={22} color="#2563eb" />
            Multi-Criteria Deployment Recommendations (MCDA)
          </h2>
          <p className="saas-card-subtitle">
            Weighted multi-criteria decision analysis across accuracy, latency, memory, and model size trade-offs.
          </p>
        </div>
      </div>

      {/* Preset & Constraint Configuration Form */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        {/* Priority Presets */}
        <div style={{ background: '#f8fafc', padding: '1.25rem', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem', display: 'block' }}>
            1. Select Operational Priority Preset
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {presets.map((p) => {
              const isSel = preset === p.id;
              return (
                <button
                  key={p.id}
                  id={`preset-${p.id}`}
                  type="button"
                  onClick={() => handlePresetSelect(p.id)}
                  style={{
                    background: isSel ? '#eff6ff' : '#ffffff',
                    border: `1.5px solid ${isSel ? '#2563eb' : '#cbd5e1'}`,
                    borderRadius: '8px',
                    padding: '0.5rem 0.75rem',
                    textAlign: 'left',
                    cursor: 'pointer',
                    transition: 'all 150ms ease',
                    boxShadow: isSel ? '0 1px 3px rgba(37, 99, 235, 0.15)' : 'none',
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: '0.825rem', color: isSel ? '#1d4ed8' : '#0f172a' }}>
                    {p.label}
                  </div>
                  <div style={{ fontSize: '0.725rem', color: '#64748b' }}>
                    {p.desc}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Custom Weight Sliders if Selected */}
          {preset === 'custom' && (
            <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', marginBottom: '0.5rem' }}>
                Adjust Criteria Importance Weights (0.0 to 1.0):
              </div>
              {Object.keys(weights).map((k) => (
                <div key={k} style={{ marginBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#475569' }}>
                    <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{k.replace('_', ' ')}:</span>
                    <span className="font-mono">{Math.round((weights[k] || 0) * 100)}%</span>
                  </div>
                  <input
                    id={`slider-weight-${k}`}
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={weights[k]}
                    onChange={(e) => handleWeightChange(k, parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#2563eb' }}
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Hardware Constraints Form */}
        <div style={{ background: '#f8fafc', padding: '1.25rem', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem', display: 'block' }}>
            2. Deployment Hardware Constraints (Optional Thresholds)
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem' }}>
            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.25rem' }}>
                Max Latency (ms)
              </label>
              <input
                id="input-constraint-latency"
                type="number"
                placeholder="e.g. 50"
                value={constraints.maxLatency}
                onChange={(e) => handleConstraintChange('maxLatency', e.target.value)}
                className="saas-input"
              />
            </div>
            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.25rem' }}>
                Max Memory (MiB)
              </label>
              <input
                id="input-constraint-memory"
                type="number"
                placeholder="e.g. 150"
                value={constraints.maxMemory}
                onChange={(e) => handleConstraintChange('maxMemory', e.target.value)}
                className="saas-input"
              />
            </div>
            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.25rem' }}>
                Max Artifact Size (MB)
              </label>
              <input
                id="input-constraint-size"
                type="number"
                placeholder="e.g. 25"
                value={constraints.maxSize}
                onChange={(e) => handleConstraintChange('maxSize', e.target.value)}
                className="saas-input"
              />
            </div>
            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.25rem' }}>
                Min Accuracy (%)
              </label>
              <input
                id="input-constraint-accuracy"
                type="number"
                placeholder="e.g. 70"
                value={constraints.minAccuracy}
                onChange={(e) => handleConstraintChange('minAccuracy', e.target.value)}
                className="saas-input"
              />
            </div>
          </div>

          <div style={{ marginTop: '1.25rem' }}>
            <button
              id="btn-generate-recommendation"
              onClick={handleGenerate}
              disabled={loading}
              className="btn btn-primary"
              style={{ fontWeight: 600, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
            >
              <Zap size={15} />
              {loading ? 'Evaluating MCDA...' : 'Generate Recommendation'}
            </button>
          </div>
        </div>
      </div>

      {/* Inline Validation Error Message */}
      {validationError && (
        <div
          id="recommendation-validation-error"
          style={{
            padding: '0.85rem 1rem',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            color: '#b91c1c',
            marginBottom: '1.25rem',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <AlertTriangle size={16} />
          <span>{validationError}</span>
        </div>
      )}

      {/* Backend API Error Message */}
      {errorMsg && (
        <div
          style={{
            padding: '0.85rem 1rem',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            color: '#b91c1c',
            marginBottom: '1.25rem',
            fontSize: '0.85rem',
          }}
        >
          <strong>Error: </strong> {errorMsg}
        </div>
      )}

      {/* Loading State Spinner */}
      {loading && (
        <div
          id="recommendations-loading-state"
          style={{
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            padding: '2.5rem',
            textAlign: 'center',
            marginBottom: '1.5rem',
            color: '#64748b',
          }}
        >
          <RefreshCw size={28} className="animate-spin" style={{ margin: '0 auto 0.75rem', color: '#2563eb' }} />
          <div style={{ fontWeight: 600, color: '#0f172a', fontSize: '1rem' }}>
            Evaluating Multi-Criteria Decision Analysis...
          </div>
          <div style={{ fontSize: '0.825rem', color: '#64748b', marginTop: '0.25rem' }}>
            Normalizing performance metrics, applying hardware constraints, and ranking eligible deployment candidates.
          </div>
        </div>
      )}

      {/* Stale State Banner (shown when inputs change after generation) */}
      {!loading && hasGenerated && isStale && (
        <div
          id="recommendations-stale-banner"
          style={{
            background: '#fffbeb',
            border: '1px solid #fde68a',
            borderRadius: '10px',
            padding: '1rem 1.25rem',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <AlertTriangle size={20} color="#d97706" />
            <div>
              <strong style={{ color: '#92400e', fontSize: '0.9rem' }}>Configuration Modified — Recommendations Stale</strong>
              <p style={{ color: '#b45309', fontSize: '0.825rem', margin: '0.2rem 0 0' }}>
                Requirements have changed since recommendations were generated. Click <strong>Generate Recommendation</strong> to view updated rankings.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleGenerate}
            className="btn btn-primary btn-sm"
            style={{ fontWeight: 600, whiteSpace: 'nowrap' }}
          >
            Regenerate Recommendations
          </button>
        </div>
      )}

      {/* Unprompted Initial Empty State (before first generation) */}
      {!loading && !hasGenerated && (
        <div
          id="recommendations-unprompted-state"
          style={{
            background: '#f8fafc',
            border: '1px dashed #cbd5e1',
            borderRadius: '12px',
            padding: '2.5rem 1.5rem',
            textAlign: 'center',
            color: '#64748b',
            marginBottom: '1.5rem',
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
              margin: '0 auto 1rem',
              color: '#2563eb',
            }}
          >
            <Sliders size={24} />
          </div>
          <h3 style={{ color: '#0f172a', fontSize: '1.1rem', marginBottom: '0.35rem', fontWeight: 700 }}>
            Configure Requirements and Generate Recommendations
          </h3>
          <p style={{ color: '#64748b', fontSize: '0.85rem', maxWidth: '540px', lineHeight: '1.5', margin: '0 auto' }}>
            Select an operational priority preset, adjust criteria weights, or specify edge hardware thresholds above, then click <strong>Generate Recommendation</strong> to rank the top vision models.
          </p>
        </div>
      )}

      {/* Recommendation Results (only shown when generated and not stale) */}
      {!loading && hasGenerated && !isStale && recData && (
        <div>
          {topRecommendations.length > 0 ? (
            <div>
              {/* Notice if fewer than three candidates qualify */}
              {topRecommendations.length < 3 && (
                <div
                  id="fewer-than-three-notice"
                  style={{
                    background: '#f0fdf4',
                    border: '1px solid #bbf7d0',
                    borderRadius: '8px',
                    padding: '0.75rem 1rem',
                    marginBottom: '1rem',
                    fontSize: '0.825rem',
                    color: '#166534',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                  }}
                >
                  <Info size={16} color="#16a34a" />
                  <span>
                    Showing all <strong>{topRecommendations.length}</strong> eligible candidate{topRecommendations.length > 1 ? 's' : ''} satisfying your specified constraints. Remaining candidates did not meet the hardware thresholds and were excluded.
                  </span>
                </div>
              )}

              {/* Top 3 Recommendation Cards */}
              <div
                id="top-recommendations-list"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1.25rem',
                  marginBottom: '1.75rem',
                }}
              >
                {topRecommendations.map((cand, idx) => {
                  const colorScheme = CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length];
                  const isTopMatch = idx === 0;

                  return (
                    <div
                      key={`${cand.model_id}-${cand.runtime}`}
                      id={`recommendation-card-${idx + 1}`}
                      style={{
                        background: '#ffffff',
                        border: `2px solid ${isTopMatch ? '#2563eb' : colorScheme.border}`,
                        borderRadius: '12px',
                        padding: '1.5rem',
                        boxShadow: isTopMatch
                          ? '0 4px 14px rgba(37, 99, 235, 0.1)'
                          : '0 2px 6px rgba(0, 0, 0, 0.03)',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'flex-start',
                          flexWrap: 'wrap',
                          gap: '1rem',
                          marginBottom: '1rem',
                        }}
                      >
                        <div>
                          <span
                            className={isTopMatch ? 'badge-live' : 'badge'}
                            style={{
                              marginBottom: '0.5rem',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              background: isTopMatch ? undefined : colorScheme.light,
                              color: isTopMatch ? undefined : colorScheme.badge,
                              border: isTopMatch ? undefined : `1px solid ${colorScheme.border}`,
                              fontWeight: 700,
                            }}
                          >
                            <Award size={13} />
                            {colorScheme.label}
                          </span>
                          <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: '#0f172a', margin: '0.2rem 0' }}>
                            {cand.display_name} [{cand.runtime?.toUpperCase()}]
                          </h3>
                          <div style={{ fontSize: '0.825rem', color: '#64748b' }}>
                            Model: <code className="font-mono">{cand.model_id}</code> • Standardized Host CPU Execution
                          </div>
                        </div>

                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Composite Score</div>
                          <div
                            className="font-mono"
                            style={{
                              fontSize: '1.85rem',
                              fontWeight: 800,
                              color: colorScheme.primary,
                              lineHeight: 1.1,
                            }}
                          >
                            {cand.composite_score != null ? cand.composite_score.toFixed(1) : 'N/A'}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Out of 100.0</div>
                        </div>
                      </div>

                      {/* Metrics Grid */}
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                          gap: '0.75rem',
                          marginBottom: '1rem',
                        }}
                      >
                        <div style={{ background: '#f8fafc', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Top-1 Accuracy</div>
                          <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
                            {cand.raw_metrics?.accuracy != null
                              ? `${(cand.raw_metrics.accuracy * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                        </div>
                        <div style={{ background: '#f8fafc', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Mean Latency</div>
                          <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
                            {cand.raw_metrics?.latency != null
                              ? `${cand.raw_metrics.latency.toFixed(2)} ms`
                              : 'N/A'}
                          </div>
                        </div>
                        <div style={{ background: '#f8fafc', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>RAM Footprint</div>
                          <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
                            {cand.raw_metrics?.memory != null
                              ? `${cand.raw_metrics.memory.toFixed(1)} MiB`
                              : 'N/A'}
                          </div>
                        </div>
                        <div style={{ background: '#f8fafc', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Model Size</div>
                          <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
                            {cand.raw_metrics?.model_size != null
                              ? `${cand.raw_metrics.model_size.toFixed(2)} MB`
                              : 'N/A'}
                          </div>
                        </div>
                      </div>

                      {/* Qualitative Rationale */}
                      <div
                        style={{
                          background: '#f8fafc',
                          borderRadius: '8px',
                          padding: '0.85rem 1rem',
                          marginBottom: '0.65rem',
                          fontSize: '0.85rem',
                          lineHeight: '1.5',
                          color: '#1e293b',
                          border: '1px solid #e2e8f0',
                        }}
                      >
                        <strong>Priority Rationale: </strong> {getCandidateRationale(cand, idx, recData?.criteria_weights_applied || weights)}
                      </div>

                      {/* Satisfied Constraints */}
                      <div
                        style={{
                          background: '#f8fafc',
                          borderRadius: '8px',
                          padding: '0.75rem 1rem',
                          marginBottom: '0.65rem',
                          fontSize: '0.825rem',
                          border: '1px solid #e2e8f0',
                        }}
                      >
                        <strong style={{ color: '#0f172a', marginRight: '0.5rem' }}>Satisfied Constraints: </strong>
                        {cand.satisfied_constraints && cand.satisfied_constraints.length > 0 ? (
                          <div style={{ display: 'inline-flex', flexWrap: 'wrap', gap: '0.35rem', marginTop: '0.25rem' }}>
                            {cand.satisfied_constraints.map((sc, scIdx) => (
                              <span
                                key={scIdx}
                                className="badge badge-completed"
                                style={{ fontSize: '0.72rem', display: 'inline-flex', alignItems: 'center', gap: '0.2rem' }}
                              >
                                <Check size={11} /> {sc}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span style={{ color: '#64748b', fontSize: '0.8rem' }}>
                            No hardware constraints specified.
                          </span>
                        )}
                      </div>

                      {/* Trade-Off Analysis */}
                      <div
                        style={{
                          background: '#f8fafc',
                          borderRadius: '8px',
                          padding: '0.75rem 1rem',
                          fontSize: '0.825rem',
                          lineHeight: '1.5',
                          color: '#475569',
                          border: '1px solid #e2e8f0',
                        }}
                      >
                        <strong style={{ color: '#0f172a' }}>Trade-Offs & Confidence: </strong>{' '}
                        {getCandidateTradeOffs(cand, idx, recData)}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Compact Visual Comparison of Top Candidates */}
              <RecommendationComparisonVisual candidates={topRecommendations} />
            </div>
          ) : recData.rankings && recData.rankings.length > 0 ? (
            /* All Candidates Disqualified Banner */
            <div
              id="no-eligible-model-banner"
              style={{
                background: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '12px',
                padding: '1.5rem',
                marginBottom: '1.5rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <XCircle size={22} color="#dc2626" />
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#991b1b', margin: 0 }}>
                  No Eligible Model Found
                </h3>
              </div>
              <p style={{ color: '#b91c1c', fontSize: '0.875rem', lineHeight: '1.5', margin: '0 0 0.5rem' }}>
                {recData.explanation}
              </p>
              <div style={{ fontSize: '0.825rem', color: '#7f1d1d' }}>
                <strong>Recommendation: </strong> {recData.trade_off_analysis}
              </div>
            </div>
          ) : (
            /* Neutral Empty State: No Benchmark Data */
            <div
              id="recommendations-empty-state"
              style={{
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '12px',
                marginBottom: '1.5rem',
                textAlign: 'center',
                padding: '2.5rem 1.75rem',
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
                <Database size={24} />
              </div>
              <h3 style={{ color: '#0f172a', fontSize: '1.15rem', marginBottom: '0.4rem', fontWeight: 700 }}>
                No Benchmark Data Available
              </h3>
              <p style={{ color: '#64748b', fontSize: '0.85rem', maxWidth: '580px', lineHeight: '1.5', margin: '0 auto 1.25rem' }}>
                {recData.explanation ||
                  'No benchmark measurements are currently available in the database. Run benchmarks from the Benchmark tab to generate deployment recommendations.'}
              </p>
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '0.75rem 1rem', maxWidth: '580px', margin: '0 auto 1.25rem', fontSize: '0.8rem', color: '#475569', textAlign: 'left' }}>
                <strong>Host CPU Requirement: </strong> The Multi-Criteria Decision Analysis (MCDA) engine requires completed <strong>Host CPU live execution</strong> measurements to calculate trade-off rankings. Edge Impulse analytical estimates are theoretical projections and cannot be ranked as physical measurements.
              </div>
              {onNavigate && (
                <button
                  id="btn-navigate-benchmark-from-rec"
                  onClick={() => onNavigate('benchmark')}
                  className="btn btn-primary btn-sm"
                  style={{ fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
                >
                  <Play size={14} fill="currentColor" />
                  Run Benchmark on Host CPU
                </button>
              )}
            </div>
          )}

          {/* Full Candidate Rankings Table if Results Exist */}
          {recData.rankings && recData.rankings.length > 0 && (
            <div style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem', color: '#0f172a' }}>
                All Evaluated Candidate Rankings
              </h3>
              <div className="saas-table-container">
                <table className="saas-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Model Configuration</th>
                      <th>Runtime</th>
                      <th>Composite Score</th>
                      <th>Accuracy</th>
                      <th>Latency</th>
                      <th>Memory</th>
                      <th>Model Size</th>
                      <th>Eligibility</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recData.rankings.map((c) => (
                      <tr key={`${c.model_id}-${c.runtime}`} style={{ opacity: c.eligible ? 1 : 0.65 }}>
                        <td>
                          <span className="font-mono" style={{ fontWeight: 700, color: c.eligible ? '#2563eb' : '#94a3b8' }}>
                            #{c.rank}
                          </span>
                        </td>
                        <td>
                          <div style={{ fontWeight: 600 }}>{c.display_name}</div>
                          <div style={{ fontSize: '0.725rem', color: '#64748b' }}>{c.model_id}</div>
                        </td>
                        <td>
                          <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a' }}>
                            {c.runtime ? c.runtime.toUpperCase() : 'N/A'}
                          </span>
                        </td>
                        <td>
                          <span className="font-mono" style={{ fontWeight: 700, color: c.eligible ? '#0284c7' : '#94a3b8' }}>
                            {c.composite_score != null ? c.composite_score.toFixed(1) : 'N/A'}
                          </span>
                        </td>
                        <td className="font-mono">
                          {c.raw_metrics?.accuracy != null ? `${(c.raw_metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
                        </td>
                        <td className="font-mono">
                          {c.raw_metrics?.latency != null ? `${c.raw_metrics.latency.toFixed(2)} ms` : 'N/A'}
                        </td>
                        <td className="font-mono">
                          {c.raw_metrics?.memory != null ? `${c.raw_metrics.memory.toFixed(1)} MiB` : 'N/A'}
                        </td>
                        <td className="font-mono">
                          {c.raw_metrics?.model_size != null ? `${c.raw_metrics.model_size.toFixed(2)} MB` : 'N/A'}
                        </td>
                        <td>
                          {c.eligible ? (
                            <span className="badge badge-completed">Eligible</span>
                          ) : (
                            <span
                              className="badge badge-failed"
                              title={Array.isArray(c.disqualification_reasons) ? c.disqualification_reasons.join('; ') : ''}
                            >
                              Disqualified
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
