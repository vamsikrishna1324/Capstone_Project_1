import React from 'react';
import {
  Check,
  CheckSquare,
  Square,
  Box,
  HardDrive,
  Cpu,
  BarChart2,
  AlertCircle,
  RefreshCw,
  PlusCircle,
  Play,
  Layers,
  ArrowRight,
  Zap,
} from 'lucide-react';

export default function ModelSelector({
  models = [],
  savedResults,
  selectedConfigs = {}, // { 'mobilenet_v3_small': ['onnx', 'tflite'], ... }
  onChangeConfigs,
  onInitiateComparison,
  hasActiveComparison = false,
  isSelectionStale = false,
  onNavigate,
}) {
  // Build lookup map for model metadata by model_id
  const modelsMap = {};
  models.forEach((m) => {
    modelsMap[m.model_id] = m;
  });

  // --------------------------------------------------------------------------
  // Mode A: savedResults is provided -> Render individual cards for saved configs
  // --------------------------------------------------------------------------
  let eligibleConfigs = [];

  if (savedResults !== undefined) {
    (savedResults || []).forEach((m, idx) => {
      // Eligible criteria: completed, not failed, not pending, not running, has valid model_id and runtime
      const isCompleted =
        (m.status === 'completed' || (!m.status && (m.latency?.mean_ms != null || m.latency_mean_ms != null))) &&
        m.status !== 'failed' &&
        m.status !== 'pending' &&
        m.status !== 'running';

      if (isCompleted && m.model_id && m.runtime) {
        const mod = modelsMap[m.model_id];
        const device = m.device || 'cpu';
        const isEstimate = m.provenance === 'edge_impulse_estimation';

        // Find artifact size from measurement or model registry
        const artifactSize =
          m.file_size_mb != null
            ? m.file_size_mb
            : m.model_size?.file_size_mb != null
            ? m.model_size.file_size_mb
            : mod?.artifacts?.[m.runtime]?.file_size_mb;

        eligibleConfigs.push({
          cardKey: m.id || `${m.run_id}-${m.model_id}-${m.runtime}-${device}-${idx}`,
          measurementId: m.id,
          run_id: m.run_id,
          model_id: m.model_id,
          display_name: mod?.display_name || m.architecture || m.model_id,
          architecture: mod?.architecture_name || m.architecture || m.model_id,
          num_classes: mod?.num_classes || 1000,
          input_dim: mod?.expected_input_size ? `${mod.expected_input_size[1]}×${mod.expected_input_size[2]}` : '224×224',
          runtime: m.runtime,
          device: device,
          provenance: m.provenance,
          isEstimate: isEstimate,
          latency: m.latency_mean_ms != null ? m.latency_mean_ms : m.latency?.mean_ms,
          top1: m.top1_accuracy != null ? m.top1_accuracy : m.accuracy?.top1_accuracy,
          top5: m.top5_accuracy != null ? m.top5_accuracy : m.accuracy?.top5_accuracy,
          ram: m.peak_ram_mb != null ? m.peak_ram_mb : m.memory?.peak_ram_mb,
          p95: m.p95_ms != null ? m.p95_ms : m.latency?.p95_ms,
          file_size_mb: artifactSize,
        });
      }
    });
  }

  // Helper: check if a completed config has a CPU sibling with identical model_id and runtime
  const hasCpuSibling = (c) => {
    return eligibleConfigs.some(
      (other) =>
        other.model_id === c.model_id &&
        other.runtime === c.runtime &&
        other.device === 'cpu' &&
        other.cardKey !== c.cardKey
    );
  };

  const isConfigCardSelected = (c) => {
    const modelSelections = selectedConfigs[c.model_id] || [];
    if (c.device && c.device !== 'cpu') {
      const specificToken = `${c.runtime}:${c.device}`;
      return (
        modelSelections.includes(specificToken) ||
        (!hasCpuSibling(c) && modelSelections.includes(c.runtime))
      );
    }
    return modelSelections.includes(c.runtime);
  };

  const toggleConfigCard = (c) => {
    const nextConfigs = { ...selectedConfigs };
    const current = [...(nextConfigs[c.model_id] || [])];
    const isSelected = isConfigCardSelected(c);
    const token = c.device && c.device !== 'cpu' ? `${c.runtime}:${c.device}` : c.runtime;

    let updated;
    if (isSelected) {
      updated = current.filter((t) => t !== token && t !== c.runtime);
    } else {
      updated = [...current, token];
    }

    if (updated.length === 0) {
      delete nextConfigs[c.model_id];
    } else {
      nextConfigs[c.model_id] = updated;
    }
    onChangeConfigs(nextConfigs);
  };

  const selectAllEligible = () => {
    const next = {};
    if (savedResults !== undefined) {
      eligibleConfigs.forEach((c) => {
        const token = c.device && c.device !== 'cpu' ? `${c.runtime}:${c.device}` : c.runtime;
        if (!next[c.model_id]) next[c.model_id] = [];
        if (!next[c.model_id].includes(token)) next[c.model_id].push(token);
      });
    } else {
      models.forEach((m) => {
        next[m.model_id] = ['pytorch', 'onnx', 'tflite'];
      });
    }
    onChangeConfigs(next);
  };

  const selectRuntimeOnly = (runtime) => {
    const next = {};
    if (savedResults !== undefined) {
      eligibleConfigs.forEach((c) => {
        if (c.runtime === runtime) {
          const token = c.device && c.device !== 'cpu' ? `${c.runtime}:${c.device}` : c.runtime;
          if (!next[c.model_id]) next[c.model_id] = [];
          if (!next[c.model_id].includes(token)) next[c.model_id].push(token);
        }
      });
    } else {
      models.forEach((m) => {
        next[m.model_id] = [runtime];
      });
    }
    onChangeConfigs(next);
  };

  const clearAll = () => {
    onChangeConfigs({});
  };

  // Count total selected configurations
  const totalSelectedCount =
    savedResults !== undefined
      ? eligibleConfigs.filter((c) => isConfigCardSelected(c)).length
      : Object.values(selectedConfigs).reduce(
          (acc, list) => acc + (list ? list.length : 0),
          0
        );

  // Fallback toggle for model-level unit test (savedResults is undefined)
  const toggleModelFallback = (modelId) => {
    const current = selectedConfigs[modelId] || [];
    const nextConfigs = { ...selectedConfigs };
    if (current.length > 0) {
      delete nextConfigs[modelId];
    } else {
      nextConfigs[modelId] = ['pytorch', 'onnx', 'tflite'];
    }
    onChangeConfigs(nextConfigs);
  };

  const toggleRuntimeFallback = (modelId, runtime) => {
    const current = selectedConfigs[modelId] || [];
    const nextConfigs = { ...selectedConfigs };
    if (current.includes(runtime)) {
      const updated = current.filter((r) => r !== runtime);
      if (updated.length === 0) delete nextConfigs[modelId];
      else nextConfigs[modelId] = updated;
    } else {
      nextConfigs[modelId] = [...current, runtime];
    }
    onChangeConfigs(nextConfigs);
  };

  // --------------------------------------------------------------------------
  // Empty State: No completed saved configurations exist in MySQL
  // --------------------------------------------------------------------------
  if (savedResults !== undefined && eligibleConfigs.length === 0) {
    return (
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
          No Completed Saved Benchmark Results Available
        </h3>
        <p style={{ color: '#64748b', fontSize: '0.85rem', maxWidth: '520px', lineHeight: '1.5', margin: '0 auto 1.25rem' }}>
          Model Comparison only evaluates configurations that have completed, saved benchmark results in MySQL. Excludes pending, failed, incomplete, or unsaved configurations. Visit <strong>Run Benchmark</strong> to evaluate models.
        </p>
        {onNavigate && (
          <button
            id="btn-navigate-to-benchmark"
            onClick={() => onNavigate('benchmark')}
            className="btn btn-primary btn-sm"
            style={{ fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
          >
            <Play size={14} fill="currentColor" />
            Go to Run Benchmark
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="saas-card" style={{ margin: '0 0 1.5rem', padding: '1.75rem' }}>
      <div className="saas-card-header">
        <div>
          <h2 className="saas-card-title">
            <Box size={22} color="#2563eb" />
            Model Comparison — Saved Results Selection Matrix
          </h2>
          <p className="saas-card-subtitle">
            Showing individual cards for every eligible completed saved benchmark configuration. Choose configurations, then click Compare Selected.
          </p>
        </div>

        {/* Quick Selection Shortcuts and Comparison Trigger */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span className="badge font-mono" style={{ padding: '0.35rem 0.75rem', background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
            {totalSelectedCount} Configs Selected
          </span>

          {onInitiateComparison && (
            <button
              id="btn-compare-selected"
              onClick={onInitiateComparison}
              disabled={totalSelectedCount === 0}
              className={`btn btn-sm ${isSelectionStale ? 'btn-warning' : 'btn-primary'}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontWeight: 600,
                padding: '0.45rem 0.95rem',
                opacity: totalSelectedCount === 0 ? 0.5 : 1,
                cursor: totalSelectedCount === 0 ? 'not-allowed' : 'pointer',
              }}
            >
              {isSelectionStale ? <RefreshCw size={14} /> : <BarChart2 size={14} />}
              {isSelectionStale ? 'Update Comparison' : (totalSelectedCount > 0 ? `Compare Selected (${totalSelectedCount})` : 'Compare Selected')}
            </button>
          )}

          <button id="btn-select-all" onClick={selectAllEligible} className="btn btn-secondary btn-sm">
            Select All Eligible
          </button>
          <button id="btn-select-onnx" onClick={() => selectRuntimeOnly('onnx')} className="btn btn-secondary btn-sm">
            All ONNX
          </button>
          <button id="btn-select-tflite" onClick={() => selectRuntimeOnly('tflite')} className="btn btn-secondary btn-sm">
            All TFLite
          </button>
          <button id="btn-clear-selection" onClick={clearAll} className="btn btn-secondary btn-sm" style={{ color: '#64748b' }}>
            Clear Selection
          </button>
        </div>
      </div>

      {/* Stale Selection Alert Banner */}
      {isSelectionStale && hasActiveComparison && (
        <div
          id="selection-stale-banner"
          style={{
            marginBottom: '1rem',
            padding: '0.65rem 1rem',
            borderRadius: '8px',
            background: '#fffbeb',
            border: '1px solid #fde68a',
            color: '#b45309',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.75rem',
            fontSize: '0.825rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={16} style={{ flexShrink: 0 }} />
            <span>Selection changed — comparison results below reflect previous selection</span>
          </div>
          {onInitiateComparison && (
            <button
              onClick={onInitiateComparison}
              className="btn btn-sm"
              style={{
                background: '#fef3c7',
                border: '1px solid #f59e0b',
                color: '#b45309',
                fontSize: '0.75rem',
                fontWeight: 600,
                padding: '0.25rem 0.6rem',
              }}
            >
              Update Comparison
            </button>
          )}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* View 1: When savedResults is provided -> Individual Config Cards   */}
      {/* ------------------------------------------------------------------ */}
      {savedResults !== undefined ? (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.775rem', fontWeight: 600, color: '#64748b' }}>
              Models with Saved Results:
            </span>
            {Array.from(new Set(eligibleConfigs.map((c) => c.model_id))).map((mId) => {
              const mod = modelsMap[mId] || { model_id: mId, display_name: mId };
              return (
                <span
                  key={mId}
                  className="badge font-mono"
                  style={{
                    background: '#f1f5f9',
                    color: '#0f172a',
                    padding: '0.25rem 0.6rem',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    border: '1px solid #cbd5e1',
                  }}
                >
                  {mod.display_name}
                </span>
              );
            })}
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(310px, 1fr))',
              gap: '1rem',
            }}
          >
            {eligibleConfigs.map((cfg) => {
              const isSelected = isConfigCardSelected(cfg);
              const runtimeLabel =
                cfg.runtime === 'onnx'
                  ? 'ONNX Runtime (.onnx)'
                  : cfg.runtime === 'tflite'
                  ? 'TensorFlow Lite (.tflite)'
                  : 'PyTorch (CPU)';

              const deviceLabel =
                cfg.device === 'cpu'
                  ? 'Host CPU'
                  : cfg.device === 'raspberry-pi-4'
                  ? 'Raspberry Pi 4'
                  : cfg.device;

              // Generate stable checkbox ID
              // For CPU or singletons, keep check-${model_id}-${runtime} for test compatibility
              const hasSibling = hasCpuSibling(cfg);
              const checkId =
                cfg.device !== 'cpu' && hasSibling
                  ? `check-${cfg.model_id}-${cfg.runtime}-${cfg.device}`
                  : `check-${cfg.model_id}-${cfg.runtime}`;

              return (
                <div
                  key={cfg.cardKey}
                  id={`config-card-${cfg.cardKey}`}
                  style={{
                    background: isSelected ? '#eff6ff' : '#ffffff',
                    border: `2px solid ${isSelected ? '#2563eb' : '#e2e8f0'}`,
                    borderRadius: '10px',
                    padding: '1.25rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '0.85rem',
                    transition: 'all 150ms ease',
                    boxShadow: isSelected ? '0 0 0 3px rgba(37, 99, 235, 0.12)' : 'none',
                  }}
                >
                  {/* Card Header with Checkbox, Title & Provenance */}
                  <div>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.65rem' }}>
                        <label
                          htmlFor={checkId}
                          style={{
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            marginTop: '2px',
                          }}
                        >
                          <input
                            type="checkbox"
                            id={checkId}
                            checked={isSelected}
                            onChange={() => toggleConfigCard(cfg)}
                            style={{ accentColor: '#2563eb', width: '16px', height: '16px', cursor: 'pointer' }}
                          />
                        </label>
                        <div>
                          <h3 style={{ fontSize: '0.975rem', color: isSelected ? '#1d4ed8' : '#0f172a', fontWeight: 700, margin: 0 }}>
                            {cfg.display_name} [{cfg.runtime.toUpperCase()}]
                          </h3>
                          <div style={{ fontSize: '0.725rem', color: '#64748b', marginTop: '0.15rem' }}>
                            {cfg.architecture} • {cfg.num_classes} classes • {cfg.input_dim}
                          </div>
                        </div>
                      </div>

                    <span className="badge font-mono" style={{ fontSize: '0.7rem', background: '#f1f5f9', color: '#334155' }}>
                      Session #{cfg.run_id}
                    </span>
                  </div>

                  {/* Runtime, Device and Provenance Badges */}
                  <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', alignItems: 'center', marginTop: '0.5rem' }}>
                    <label
                      htmlFor={checkId}
                      style={{
                        cursor: 'pointer',
                        fontSize: '0.75rem',
                        fontWeight: 600,
                        color: '#0f172a',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                      }}
                    >
                      <span className="badge font-mono" style={{ background: '#e0f2fe', color: '#0369a1', border: '1px solid #bae6fd' }}>
                        {cfg.runtime.toUpperCase()}
                      </span>
                      <span>{runtimeLabel}</span>
                    </label>

                    <span className="badge font-mono" style={{ background: '#f8fafc', color: '#475569', border: '1px solid #e2e8f0' }}>
                      {deviceLabel}
                    </span>

                    {cfg.isEstimate ? (
                      <span className="badge-estimate" style={{ fontSize: '0.68rem' }}>Analytical Estimate</span>
                    ) : (
                      <span className="badge-live" style={{ fontSize: '0.68rem' }}>Measured Result</span>
                    )}

                    <span className="badge badge-completed" style={{ fontSize: '0.68rem' }}>Completed</span>
                  </div>
                </div>

                {/* Metrics Grid */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(3, 1fr)',
                    gap: '0.5rem',
                    background: '#f8fafc',
                    padding: '0.65rem',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    fontSize: '0.75rem',
                  }}
                >
                  <div>
                    <div style={{ color: '#64748b', fontSize: '0.675rem' }}>Latency</div>
                    <div className="font-mono" style={{ fontWeight: 700, color: '#0f172a' }}>
                      {cfg.latency != null
                        ? cfg.latency >= 1000
                          ? `${cfg.isEstimate ? '~' : ''}${(cfg.latency / 1000).toFixed(2)} s`
                          : `${cfg.isEstimate ? '~' : ''}${cfg.latency.toFixed(2)} ms`
                        : 'N/A'}
                    </div>
                  </div>

                  <div>
                    <div style={{ color: '#64748b', fontSize: '0.675rem' }}>Top-1 Accuracy</div>
                    <div className="font-mono" style={{ fontWeight: 700, color: '#0f172a' }}>
                      {cfg.top1 != null ? `${(cfg.top1 * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>

                  <div>
                    <div style={{ color: '#64748b', fontSize: '0.675rem' }}>Top-5 Accuracy</div>
                    <div className="font-mono" style={{ fontWeight: 700, color: '#0f172a' }}>
                      {cfg.top5 != null ? `${(cfg.top5 * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>

                  <div>
                    <div style={{ color: '#64748b', fontSize: '0.675rem' }}>Artifact Size</div>
                    <div className="font-mono" style={{ fontWeight: 700, color: '#0f172a' }}>
                      {cfg.file_size_mb != null ? `${cfg.file_size_mb} MB` : 'N/A'}
                    </div>
                  </div>

                  <div>
                    <div style={{ color: '#64748b', fontSize: '0.675rem' }}>RAM Usage</div>
                    <div className="font-mono" style={{ fontWeight: 700, color: '#0f172a' }}>
                      {cfg.isEstimate ? 'N/A' : (cfg.ram != null ? `${cfg.ram.toFixed(1)} MiB` : 'N/A')}
                    </div>
                  </div>

                  <div>
                    <div style={{ color: '#64748b', fontSize: '0.675rem' }}>P95 Latency</div>
                    <div className="font-mono" style={{ fontWeight: 700, color: '#0f172a' }}>
                      {cfg.isEstimate ? 'N/A' : (cfg.p95 != null ? `${cfg.p95.toFixed(2)} ms` : 'N/A')}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        </>
      ) : (
        /* ------------------------------------------------------------------ */
        /* View 2: Standalone Fallback Grid (when savedResults is undefined)   */
        /* Preserves unit test coverage for ModelSelector.test.jsx            */
        /* ------------------------------------------------------------------ */
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1rem',
          }}
        >
          {models.map((model) => {
            const selectedList = selectedConfigs[model.model_id] || [];
            const isModelSelected = selectedList.length > 0;
            const availableRts = ['pytorch', 'onnx', 'tflite'];

            return (
              <div
                key={model.model_id}
                id={`model-card-${model.model_id}`}
                style={{
                  background: isModelSelected ? '#eff6ff' : '#ffffff',
                  border: `2px solid ${isModelSelected ? '#2563eb' : '#e2e8f0'}`,
                  borderRadius: '10px',
                  padding: '1.25rem',
                  transition: 'all 150ms ease',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <button
                      id={`checkbox-model-${model.model_id}`}
                      onClick={() => toggleModelFallback(model.model_id)}
                      aria-label={`Toggle model ${model.display_name}`}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: isModelSelected ? '#2563eb' : '#94a3b8',
                        display: 'flex',
                        alignItems: 'center',
                        padding: 0,
                      }}
                    >
                      {isModelSelected ? <CheckSquare size={20} /> : <Square size={20} />}
                    </button>
                    <div>
                      <h3 style={{ fontSize: '1rem', color: '#0f172a', fontWeight: 700 }}>{model.display_name}</h3>
                      <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                        {model.architecture_name} • {model.num_classes} classes
                      </span>
                    </div>
                  </div>
                  <span className="badge font-mono" style={{ fontSize: '0.7rem', background: '#f1f5f9', color: '#475569' }}>
                    224×224
                  </span>
                </div>

                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                    marginTop: '0.85rem',
                    paddingTop: '0.75rem',
                    borderTop: '1px solid #f1f5f9',
                  }}
                >
                  {availableRts.map((rt) => {
                    const isChecked = selectedList.includes(rt);
                    const artifactSizeMb =
                      rt === 'onnx'
                        ? model.artifacts?.onnx?.file_size_mb
                        : rt === 'tflite'
                        ? model.artifacts?.tflite?.file_size_mb
                        : null;

                    return (
                      <label
                        key={rt}
                        htmlFor={`check-${model.model_id}-${rt}`}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          fontSize: '0.825rem',
                          cursor: 'pointer',
                          padding: '0.4rem 0.6rem',
                          borderRadius: '6px',
                          background: isChecked ? '#dbeafe' : '#f8fafc',
                          border: `1px solid ${isChecked ? '#93c5fd' : '#e2e8f0'}`,
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <input
                            type="checkbox"
                            id={`check-${model.model_id}-${rt}`}
                            checked={isChecked}
                            onChange={() => toggleRuntimeFallback(model.model_id, rt)}
                            style={{ accentColor: '#2563eb' }}
                          />
                          <span style={{ fontWeight: 600, color: '#0f172a' }}>
                            {rt === 'pytorch' && 'PyTorch (CPU)'}
                            {rt === 'onnx' && 'ONNX Runtime (.onnx)'}
                            {rt === 'tflite' && 'TensorFlow Lite (.tflite)'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
                          {artifactSizeMb != null && (
                            <span className="badge font-mono" style={{ fontSize: '0.7rem', background: '#f1f5f9', color: '#475569' }}>
                              {artifactSizeMb} MB
                            </span>
                          )}
                        </div>
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
