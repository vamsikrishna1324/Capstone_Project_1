import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Settings2,
  ShieldAlert,
  CheckCircle2,
  Loader2,
  Gauge,
  AlertCircle,
  Cpu,
  Box,
  Layers,
  Database,
  RefreshCw,
  ExternalLink,
  History,
  Check,
  Zap,
} from 'lucide-react';
import AlertModal from './Common/AlertModal';
import {
  runBenchmarkComparison,
  fetchBenchmarkRun,
  fetchBenchmarkProgress,
  profileModelEdgeImpulse,
} from '../api/client';

export default function BenchmarkRunner({
  models = [],
  targets = [],
  savedResults = [],
  selectedConfigs,
  onBenchmarkSuccess,
  onNavigate,
}) {
  // If selectedConfigs is explicitly passed from parent or test:
  const configKeys = selectedConfigs ? Object.keys(selectedConfigs) : [];
  const isSelectionEmpty = selectedConfigs !== undefined && configKeys.length === 0;

  // Step 1: Exactly ONE model
  const [selectedModelId, setSelectedModelId] = useState(() => {
    if (selectedConfigs && configKeys.length > 0) return configKeys[0];
    return models.length > 0 ? models[0].model_id : 'mobilenet_v3_small';
  });

  // Step 2: Format and Target
  const [selectedTarget, setSelectedTarget] = useState('cpu');
  const [selectedRuntime, setSelectedRuntime] = useState(() => {
    if (selectedConfigs && configKeys.length > 0 && selectedConfigs[configKeys[0]]?.[0]) {
      return selectedConfigs[configKeys[0]][0];
    }
    return 'onnx';
  });

  // Synchronize when selectedConfigs changes
  useEffect(() => {
    if (selectedConfigs && configKeys.length > 0) {
      const firstModel = configKeys[0];
      setSelectedModelId(firstModel);
      if (selectedConfigs[firstModel]?.[0]) {
        setSelectedRuntime(selectedConfigs[firstModel][0]);
      }
    }
  }, [selectedConfigs]);

  // Update default when models load
  useEffect(() => {
    if (models.length > 0 && !models.some((m) => m.model_id === selectedModelId) && !selectedConfigs) {
      setSelectedModelId(models[0].model_id);
    }
  }, [models, selectedModelId, selectedConfigs]);

  // Metrics to evaluate
  const [metrics, setMetrics] = useState({
    latency: true,
    memory: true,
    model_size: true,
    accuracy: true,
  });

  const [numRuns, setNumRuns] = useState(100);
  const [warmupRuns, setWarmupRuns] = useState(10);
  const [runName, setRunName] = useState('');
  const [reuseSavedAccuracy, setReuseSavedAccuracy] = useState(true);
  const [runAccuracyEval, setRunAccuracyEval] = useState(false);
  const [maxAccuracySamples, setMaxAccuracySamples] = useState(50);

  const [isWarningOpen, setIsWarningOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState(null);
  const [progressState, setProgressState] = useState(null);
  const [completedResult, setCompletedResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const pollingTimerRef = useRef(null);

  // Clean up polling timer on unmount
  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
      }
    };
  }, []);

  // Reset completed and status state when the user modifies configuration
  useEffect(() => {
    if (!isRunning) {
      setSuccessMsg(null);
      setErrorMsg(null);
      setCompletedResult(null);
      setProgressState(null);
    }
  }, [selectedModelId, selectedRuntime, selectedTarget]);

  // Available target options
  const targetOptions = [
    {
      id: 'cpu',
      name: 'Host CPU (Local Baseline)',
      description: 'Physical execution on local CPU baseline. Not physical edge hardware.',
      type: 'local',
      supported_formats: ['pytorch', 'onnx', 'tflite'],
    },
    ...((targets && targets.length > 0)
      ? targets.map((t) => ({
          id: t.target_id || t.id,
          name: `${t.display_name || t.name} (Edge Impulse)`,
          description: t.notes || 'Analytical estimation via Edge Impulse Studio (TFLite only).',
          type: 'edge_impulse',
          supported_formats: t.supported_formats || ['tflite'],
        }))
      : [
          {
            id: 'raspberry-pi-4',
            name: 'Raspberry Pi 4 (Edge Impulse)',
            description: 'Analytical estimation via Edge Impulse Studio (TFLite only).',
            type: 'edge_impulse',
            supported_formats: ['tflite'],
          },
        ]),
  ];

  const currentModel = models.find((m) => m.model_id === selectedModelId);

  // Available formats for current selection
  const availableRuntimes = [
    { id: 'pytorch', label: 'PyTorch' },
    { id: 'onnx', label: 'ONNX Runtime (.onnx)' },
    { id: 'tflite', label: 'TensorFlow Lite (.tflite)' },
  ];

  // Step 3: Check for matching results in savedResults
  const matchingResults = (savedResults || []).filter((m) => {
    if (m.model_id !== selectedModelId) return false;
    if (m.runtime !== selectedRuntime) return false;
    if (selectedTarget === 'cpu') {
      return m.provenance !== 'edge_impulse_estimation' && (m.device == null || m.device === 'cpu');
    } else {
      return m.provenance === 'edge_impulse_estimation' || m.device === selectedTarget;
    }
  });

  const reusableResult = matchingResults.find((m) => m.status === 'completed');
  const ineligibleResult = !reusableResult && matchingResults.length > 0 ? matchingResults[0] : null;

  const isPrecomputedAccuracyMissing = Boolean(
    selectedTarget === 'cpu' &&
    !runAccuracyEval &&
    metrics.accuracy &&
    (currentModel?.artifacts?.[selectedRuntime]?.has_precomputed_accuracy === false ||
     (selectedModelId === 'densenet121' && selectedRuntime === 'tflite') ||
     (selectedModelId === 'efficientnet_b0' && selectedRuntime === 'tflite'))
  );

  const toggleMetric = (key) => {
    setMetrics((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleAccuracyToggle = (e) => {
    const checked = e.target.checked;
    if (checked) {
      setIsWarningOpen(true);
    } else {
      setRunAccuracyEval(false);
      setReuseSavedAccuracy(true);
    }
  };

  const confirmFreshAccuracy = () => {
    setRunAccuracyEval(true);
    setReuseSavedAccuracy(false);
    setIsWarningOpen(false);
  };

  const cancelFreshAccuracy = () => {
    setRunAccuracyEval(false);
    setReuseSavedAccuracy(true);
    setIsWarningOpen(false);
  };

  // Helper to format seconds into readable string
  const formatDuration = (seconds) => {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return 'Not available yet';
    const s = Math.max(0, Math.round(seconds));
    if (s < 60) return `${s}s`;
    const mins = Math.floor(s / 60);
    const rem = s % 60;
    return `${mins}m ${rem}s`;
  };

  // Polling loop for active in-flight run
  const startPollingProgress = (runId) => {
    if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
    setActiveRunId(runId);
    let fetchRunAttempts = 0;

    const poll = async () => {
      try {
        const prog = await fetchBenchmarkProgress(runId);
        setProgressState(prog);

        if (prog.status === 'completed' || prog.status === 'partial') {
          // Retrieve final run details; do not declare success until saved run state is reliably available
          try {
            const finalRun = await fetchBenchmarkRun(runId);
            clearInterval(pollingTimerRef.current);
            pollingTimerRef.current = null;
            setIsRunning(false);

            const finalWarnings = (finalRun.warnings && finalRun.warnings.length > 0)
              ? finalRun.warnings
              : (prog.warnings && prog.warnings.length > 0 ? prog.warnings : finalRun.warnings);
            const resolvedRun = finalWarnings !== undefined
              ? { ...finalRun, warnings: finalWarnings }
              : finalRun;

            setCompletedResult(resolvedRun);
            const mList = resolvedRun.measurements || [];
            const allFailed = mList.length > 0 && mList.every((m) => m.status === 'failed');
            const anyFailed = mList.some((m) => m.status === 'failed');
            if (resolvedRun.status === 'failed' || allFailed) {
              setErrorMsg(resolvedRun.error_message || 'Benchmark measurements failed.');
            } else if (anyFailed || resolvedRun.evaluation_status === 'partial') {
              setSuccessMsg(`Benchmark completed with partial results. Run ID #${runId} recorded.`);
              if (onBenchmarkSuccess) onBenchmarkSuccess(resolvedRun);
            } else {
              setSuccessMsg(`Benchmark completed successfully! Run ID #${runId}`);
              if (onBenchmarkSuccess) onBenchmarkSuccess(resolvedRun);
            }
          } catch (fetchErr) {
            fetchRunAttempts += 1;
            // Retry fetching saved run up to 5 times (2 seconds)
            if (fetchRunAttempts >= 5) {
              clearInterval(pollingTimerRef.current);
              pollingTimerRef.current = null;
              setIsRunning(false);

              const fallbackRun = {
                id: runId,
                status: prog.status,
                warnings: prog.warnings || [],
                measurements: [],
              };
              setCompletedResult(fallbackRun);
              setErrorMsg(`Run completed, but unable to load saved run details: ${fetchErr.message || fetchErr}`);
            }
          }
        } else if (prog.status === 'failed') {
          clearInterval(pollingTimerRef.current);
          pollingTimerRef.current = null;
          setIsRunning(false);
          setErrorMsg(prog.error_message || 'Benchmark evaluation failed on the backend.');
        }
      } catch (err) {
        // If progress fails, check if the run completed in DB
        try {
          const run = await fetchBenchmarkRun(runId);
          if (run.status === 'completed' || run.status === 'partial') {
            clearInterval(pollingTimerRef.current);
            pollingTimerRef.current = null;
            setIsRunning(false);
            setCompletedResult(run);
            const mList = run.measurements || [];
            const allFailed = mList.length > 0 && mList.every((m) => m.status === 'failed');
            const anyFailed = mList.some((m) => m.status === 'failed');
            if (run.status === 'failed' || allFailed) {
              setErrorMsg(run.error_message || 'Benchmark measurements failed.');
            } else if (anyFailed || run.evaluation_status === 'partial') {
              setSuccessMsg(`Benchmark completed with partial results. Run ID #${runId} recorded.`);
              if (onBenchmarkSuccess) onBenchmarkSuccess(run);
            } else {
              setSuccessMsg(`Benchmark completed successfully! Run ID #${runId}`);
              if (onBenchmarkSuccess) onBenchmarkSuccess(run);
            }
          } else if (run.status === 'failed') {
            clearInterval(pollingTimerRef.current);
            pollingTimerRef.current = null;
            setIsRunning(false);
            setErrorMsg(run.error_message || 'Benchmark evaluation failed.');
          }
        } catch (_) {}
      }
    };

    // Run first poll immediately, then every 400ms
    poll();
    pollingTimerRef.current = setInterval(poll, 400);
  };

  // Use an existing completed result directly
  const handleUseExistingResult = (reusedItem) => {
    const warnings = [];
    const accTop1 = reusedItem.accuracy?.top1_accuracy ?? reusedItem.top1_accuracy;
    if (accTop1 === null || accTop1 === undefined) {
      const modelName = reusedItem.architecture || currentModel?.display_name || reusedItem.model_id;
      const rtLabel = (reusedItem.runtime || selectedRuntime || '').toUpperCase();
      warnings.push(
        `Verified precomputed accuracy is unavailable for ${modelName} on ${rtLabel}. Top-1 and Top-5 accuracy remain N/A unless fresh live evaluation is performed.`
      );
    }

    const completedObj = {
      id: reusedItem.run_id,
      device: reusedItem.device,
      provenance: reusedItem.provenance,
      status: 'completed',
      reused: true,
      warnings,
      measurements: [reusedItem],
    };

    setCompletedResult(completedObj);
    setSuccessMsg(`Existing saved benchmark result reused! Run ID #${reusedItem.run_id} loaded.`);
    if (onBenchmarkSuccess) {
      onBenchmarkSuccess(completedObj);
    }
  };

  // Execute benchmark
  const handleExecute = async () => {
    if (isRunning) return; // Prevent duplicate clicks

    // Validation
    if (isSelectionEmpty || !selectedModelId) {
      setErrorMsg('Please select at least one configuration before initiating benchmark.');
      return;
    }

    // Edge Impulse profiling branch
    if (selectedTarget === 'raspberry-pi-4') {
      if (selectedRuntime !== 'tflite') {
        setErrorMsg('Edge Impulse profiling on Raspberry Pi 4 supports TFLite only.');
        return;
      }

      setIsRunning(true);
      setErrorMsg(null);
      setSuccessMsg(null);
      setCompletedResult(null);
      setProgressState(null);

      const edgePayload = {
        model_id: selectedModelId,
        target_device: 'raspberry-pi-4',
        runtime: 'tflite',
        batch_size: 1,
        reuse_completed: true,
        reuse_saved_accuracy: reuseSavedAccuracy,
      };

      try {
        const result = await profileModelEdgeImpulse(edgePayload);
        setCompletedResult(result);
        setSuccessMsg(`Edge Impulse profiling completed successfully! Run ID #${result.id}`);
        if (onBenchmarkSuccess) {
          onBenchmarkSuccess(result);
        }
      } catch (err) {
        setErrorMsg(err.message || 'Edge Impulse profiling failed.');
      } finally {
        setIsRunning(false);
      }
      return;
    }

    // Host CPU live execution branch
    const activeMetrics = Object.keys(metrics).filter((k) => metrics[k]);
    if (activeMetrics.length === 0) {
      setErrorMsg('Please select at least one performance metric to measure.');
      return;
    }

    setIsRunning(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    setCompletedResult(null);
    setProgressState({
      status: 'preparing',
      images_completed: 0,
      total_images: runAccuracyEval ? Number(maxAccuracySamples) : (Number(warmupRuns) + Number(numRuns)),
      images_remaining: runAccuracyEval ? Number(maxAccuracySamples) : (Number(warmupRuns) + Number(numRuns)),
      percentage: 0.0,
      elapsed_seconds: 0.0,
      processing_speed_fps: null,
      estimated_remaining_seconds: null,
    });

    const payload = {
      model_ids: [selectedModelId],
      runtimes: [selectedRuntime],
      metrics: activeMetrics,
      num_runs: Number(numRuns),
      warmup_runs: Number(warmupRuns),
      batch_size: 1,
      device: 'cpu',
      run_name: runName.trim() || undefined,
      reuse_saved_accuracy: reuseSavedAccuracy,
      run_accuracy_eval: runAccuracyEval,
      max_accuracy_samples: runAccuracyEval ? Number(maxAccuracySamples) : undefined,
      async_run: true,
    };

    try {
      const initialRun = await runBenchmarkComparison(payload);
      if (initialRun && initialRun.status === 'running') {
        // Asynchronous execution started: start live progress polling
        startPollingProgress(initialRun.id);
      } else if (
        initialRun &&
        (initialRun.status === 'completed' ||
          initialRun.status === 'failed' ||
          !initialRun.status ||
          (initialRun.measurements && initialRun.measurements.length > 0))
      ) {
        // Immediate / synchronous return (or mocked test)
        setCompletedResult(initialRun);
        setIsRunning(false);
        const mList = initialRun.measurements || [];
        const allFailed = mList.length > 0 && mList.every((m) => m.status === 'failed');
        const anyFailed = mList.some((m) => m.status === 'failed');
        if (initialRun.status === 'failed' || allFailed) {
          setErrorMsg(initialRun.error_message || 'Benchmark measurements failed.');
        } else if (anyFailed || initialRun.evaluation_status === 'partial') {
          setSuccessMsg(`Benchmark completed with partial results. Run ID #${initialRun.id} recorded.`);
          if (onBenchmarkSuccess) onBenchmarkSuccess(initialRun);
        } else {
          setSuccessMsg(`Benchmark completed successfully! Run ID #${initialRun.id}`);
          if (onBenchmarkSuccess) onBenchmarkSuccess(initialRun);
        }
      } else if (initialRun && initialRun.id) {
        startPollingProgress(initialRun.id);
      }
    } catch (err) {
      setIsRunning(false);
      setProgressState(null);
      setErrorMsg(err.message || 'Failed to initiate benchmark execution.');
    }
  };

  return (
    <div className="saas-card" style={{ margin: '0 0 1.5rem', padding: '1.75rem' }}>
      <div className="saas-card-header">
        <div>
          <h2 className="saas-card-title">
            <Gauge size={22} color="#2563eb" />
            Single-Model Benchmark Evaluation
          </h2>
          <p className="saas-card-subtitle">
            Configure, check for reusable results, and execute real-time performance evaluation for a single model configuration.
          </p>
        </div>
      </div>

      {/* =========================================================================
          SECTION A: Select Configuration
          ========================================================================= */}
      <div id="section-a-config" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <span className="badge" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe', fontWeight: 700 }}>A</span>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>Select Configuration</h3>
        </div>

        {/* 1. Model Selection */}
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#334155', display: 'block', marginBottom: '0.5rem' }}>
            Model Selection (Single Image-Classification Model)
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
            {models.map((m) => {
              const isSelected = selectedModelId === m.model_id;
              const inputDim = m.expected_input_size ? `${m.expected_input_size[1]}×${m.expected_input_size[2]}` : '224×224';
              const artifacts = m.artifacts || {};

              return (
                <div
                  key={m.model_id}
                  id={`model-card-${m.model_id}`}
                  onClick={() => !isRunning && setSelectedModelId(m.model_id)}
                  style={{
                    background: isSelected ? '#eff6ff' : '#ffffff',
                    border: `2px solid ${isSelected ? '#2563eb' : '#e2e8f0'}`,
                    borderRadius: '10px',
                    padding: '1rem',
                    cursor: isRunning ? 'not-allowed' : 'pointer',
                    transition: 'all 150ms ease',
                    boxShadow: isSelected ? '0 0 0 3px rgba(37, 99, 235, 0.15)' : 'none',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                    <div style={{ fontWeight: 700, color: isSelected ? '#1d4ed8' : '#0f172a', fontSize: '0.95rem' }}>
                      {m.display_name}
                    </div>
                    {isSelected && <Check size={18} color="#2563eb" />}
                  </div>

                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '0.65rem' }}>
                    Input: <strong>{inputDim}</strong> • Classes: <strong>{m.num_classes || 1000}</strong>
                  </div>

                  <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                    <span className="badge font-mono" style={{ fontSize: '0.7rem', background: '#f1f5f9', color: '#475569' }}>
                      PyTorch
                    </span>
                    <span
                      className="badge font-mono"
                      style={{
                        fontSize: '0.7rem',
                        background: artifacts.onnx?.exists ? '#ecfeff' : '#f8fafc',
                        color: artifacts.onnx?.exists ? '#0891b2' : '#94a3b8',
                        border: `1px solid ${artifacts.onnx?.exists ? '#a5f3fc' : '#e2e8f0'}`,
                      }}
                    >
                      ONNX {artifacts.onnx?.exists ? '✓' : '—'}
                    </span>
                    <span
                      className="badge font-mono"
                      style={{
                        fontSize: '0.7rem',
                        background: artifacts.tflite?.exists ? '#f0fdf4' : '#f8fafc',
                        color: artifacts.tflite?.exists ? '#16a34a' : '#94a3b8',
                        border: `1px solid ${artifacts.tflite?.exists ? '#bbf7d0' : '#e2e8f0'}`,
                      }}
                    >
                      TFLite {artifacts.tflite?.exists ? '✓' : '—'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 2. Format & Target Hardware Selection */}
        <div style={{ padding: '1.25rem', background: '#f8fafc', borderRadius: '10px', border: '1px solid #e2e8f0', marginBottom: '1.25rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
            {/* Target Selector */}
            <div>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#334155', display: 'block', marginBottom: '0.5rem' }}>
                Target Environment
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {targetOptions.map((t) => {
                  const isSel = selectedTarget === t.id;
                  return (
                    <label
                      key={t.id}
                      id={`target-option-${t.id}`}
                      htmlFor={`target-radio-${t.id}`}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '0.65rem',
                        background: isSel ? '#ffffff' : '#f1f5f9',
                        border: `2px solid ${isSel ? '#2563eb' : '#cbd5e1'}`,
                        borderRadius: '8px',
                        padding: '0.65rem 0.85rem',
                        cursor: isRunning ? 'not-allowed' : 'pointer',
                        boxShadow: isSel ? '0 1px 3px rgba(37, 99, 235, 0.15)' : 'none',
                      }}
                    >
                      <input
                        type="radio"
                        id={`target-radio-${t.id}`}
                        name="target-device-group"
                        checked={isSel}
                        disabled={isRunning}
                        onChange={() => {
                          setSelectedTarget(t.id);
                        }}
                        style={{ marginTop: '0.2rem', accentColor: '#2563eb' }}
                      />
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: isSel ? '#1d4ed8' : '#1e293b' }}>
                            {t.name}
                          </span>
                          {t.type === 'local' ? (
                            <span className="badge-live">Live Host</span>
                          ) : (
                            <span className="badge-estimate">Analytical Estimate</span>
                          )}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.2rem' }}>
                          {t.description}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Format Selector */}
            <div>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#334155', display: 'block', marginBottom: '0.5rem' }}>
                Execution Format / Runtime
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {availableRuntimes.map((rt) => {
                  const isSel = selectedRuntime === rt.id;
                  return (
                    <button
                      key={rt.id}
                      id={`runtime-option-${rt.id}`}
                      type="button"
                      disabled={isRunning}
                      onClick={() => setSelectedRuntime(rt.id)}
                      style={{
                        background: isSel ? '#ffffff' : '#f1f5f9',
                        border: `2px solid ${isSel ? '#2563eb' : '#cbd5e1'}`,
                        borderRadius: '8px',
                        padding: '0.65rem 0.85rem',
                        textAlign: 'left',
                        cursor: isRunning ? 'not-allowed' : 'pointer',
                      }}
                    >
                      <div style={{ fontWeight: 700, fontSize: '0.85rem', color: isSel ? '#1d4ed8' : '#1e293b' }}>
                        {rt.label}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.2rem' }}>
                        {rt.id === 'pytorch' && 'Native PyTorch FP32 graph'}
                        {rt.id === 'onnx' && 'Open Neural Network Exchange static graph (NCHW)'}
                        {rt.id === 'tflite' && 'FlatBuffer mobile format with NHWC tensor layout'}
                      </div>
                    </button>
                  );
                })}
              </div>

              {selectedTarget === 'raspberry-pi-4' && (
                <div style={{ marginTop: '0.75rem', padding: '0.5rem 0.75rem', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '6px', fontSize: '0.75rem', color: '#92400e' }}>
                  <strong>Format Policy: </strong> Edge Impulse profiling for Raspberry Pi 4 supports TFLite flatbuffers only. ONNX and PyTorch cannot be profiled via Edge Impulse.
                </div>
              )}
            </div>
          </div>

          {/* Methodology notice */}
          <div style={{ marginTop: '1rem', padding: '0.65rem 0.85rem', background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '6px', fontSize: '0.8rem', color: '#1e40af' }}>
            <strong>Hardware Disclaimer: </strong> Host CPU measurements represent local CPU baseline execution on MacBook development hardware and do not simulate physical edge hardware (NPU/DSP). Edge Impulse projections are analytical studio estimates.
          </div>
        </div>

        {/* 3. Selected Configuration Summary */}
        <div
          id="selected-config-summary"
          style={{
            padding: '1rem 1.25rem',
            background: '#ffffff',
            border: '1px solid #bfdbfe',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
            boxShadow: '0 1px 3px rgba(37, 99, 235, 0.08)',
          }}
        >
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#2563eb', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Selected Configuration Summary
            </div>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#0f172a', marginTop: '0.2rem' }}>
              {currentModel?.display_name || selectedModelId} • {(selectedRuntime || '').toUpperCase()} on {selectedTarget === 'cpu' ? 'Host CPU (Local Baseline)' : 'Raspberry Pi 4 (Edge Impulse)'}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {selectedTarget === 'raspberry-pi-4' ? (
              <span className="badge-estimate">Analytical Estimate</span>
            ) : (
              <span className="badge-live">Measured Result</span>
            )}
            <span className="badge font-mono" style={{ background: '#f0fdf4', color: '#16a34a', border: '1px solid #bbf7d0', fontWeight: 600 }}>
              Configuration Ready
            </span>
          </div>
        </div>
      </div>

      {/* =========================================================================
          SECTION B: Existing Result Check
          ========================================================================= */}
      <div id="section-b-reuse" style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <span className="badge" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe', fontWeight: 700 }}>B</span>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>Existing Result Check</h3>
        </div>

        {reusableResult ? (
          <div
            id="reusable-result-card"
            style={{
              background: '#f0fdf4',
              border: '1px solid #86efac',
              borderRadius: '10px',
              padding: '1.25rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.85rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <CheckCircle2 size={20} color="#16a34a" />
                <strong style={{ color: '#15803d', fontSize: '0.95rem' }}>
                  Matching Completed Result Available (Run ID #{reusableResult.run_id})
                </strong>
              </div>
              <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                {reusableResult.provenance === 'edge_impulse_estimation' ? (
                  <span className="badge-estimate">Analytical Estimate</span>
                ) : (
                  <span className="badge-live">Measured Result</span>
                )}
                <span className="badge badge-completed">Saved Result Reusable</span>
              </div>
            </div>

            <p style={{ fontSize: '0.825rem', color: '#166534', margin: 0 }}>
              An existing completed evaluation matching <strong>{reusableResult.architecture || reusableResult.model_id}</strong> on <strong>{(reusableResult.runtime || '').toUpperCase()}</strong> ({reusableResult.device || 'cpu'}) was found in MySQL. You can use these saved metrics immediately without running an expensive redundant evaluation.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem' }}>
              <div style={{ background: '#ffffff', padding: '0.65rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Mean Latency</div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>
                  {reusableResult.latency_mean_ms != null ? `${reusableResult.latency_mean_ms} ms` : (reusableResult.latency?.mean_ms != null ? `${reusableResult.latency.mean_ms} ms` : 'N/A')}
                </div>
              </div>
              <div style={{ background: '#ffffff', padding: '0.65rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Throughput</div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>
                  {reusableResult.throughput_fps != null ? `${reusableResult.throughput_fps} fps` : (reusableResult.latency?.throughput_fps != null ? `${reusableResult.latency.throughput_fps} fps` : 'N/A')}
                </div>
              </div>
              <div style={{ background: '#ffffff', padding: '0.65rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Top-1 Accuracy</div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>
                  {reusableResult.top1_accuracy != null ? `${(reusableResult.top1_accuracy * 100).toFixed(1)}%` : (reusableResult.accuracy?.top1_accuracy != null ? `${(reusableResult.accuracy.top1_accuracy * 100).toFixed(1)}%` : 'N/A')}
                </div>
              </div>
              <div style={{ background: '#ffffff', padding: '0.65rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: '0.7rem', color: '#64748b' }}>RAM Footprint</div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>
                  {reusableResult.peak_ram_mb != null ? `${reusableResult.peak_ram_mb} MiB` : (reusableResult.memory?.peak_ram_mb != null ? `${reusableResult.memory.peak_ram_mb} MiB` : 'N/A')}
                </div>
              </div>
              <div style={{ background: '#ffffff', padding: '0.65rem', borderRadius: '6px', border: '1px solid #bbf7d0' }}>
                <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Artifact Size</div>
                <div style={{ fontWeight: 700, color: '#0f172a' }}>
                  {reusableResult.file_size_mb != null ? `${reusableResult.file_size_mb} MB` : (reusableResult.model_size?.file_size_mb != null ? `${reusableResult.model_size.file_size_mb} MB` : 'N/A')}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
              <button
                id="btn-use-reused-result"
                onClick={() => handleUseExistingResult(reusableResult)}
                className="btn btn-sm btn-primary"
                style={{ fontWeight: 600 }}
              >
                Use Saved Result
              </button>
            </div>
          </div>
        ) : ineligibleResult ? (
          <div
            id="ineligible-result-notice"
            style={{
              background: '#fffbeb',
              border: '1px solid #fde68a',
              borderRadius: '10px',
              padding: '1rem 1.25rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertCircle size={18} color="#d97706" />
              <strong style={{ color: '#b45309', fontSize: '0.9rem' }}>
                Existing Result Ineligible for Reuse (Run #{ineligibleResult.run_id})
              </strong>
            </div>
            <p style={{ fontSize: '0.825rem', color: '#92400e', margin: 0 }}>
              An existing record for <strong>{ineligibleResult.architecture || ineligibleResult.model_id}</strong> on <strong>{(ineligibleResult.runtime || '').toUpperCase()}</strong> was found in MySQL, but it is <strong>ineligible for reuse</strong> because its measurement status is <span className="badge badge-failed" style={{ fontSize: '0.75rem' }}>{(ineligibleResult.status || 'unknown').toUpperCase()}</span>{ineligibleResult.error_message ? `: ${ineligibleResult.error_message}` : '.'}
            </p>
            <div style={{ fontSize: '0.8rem', color: '#78350f' }}>
              No matching completed result currently exists in MySQL for this configuration. Failed or incomplete measurements cannot be reused. Click <strong>Run Benchmark</strong> in Section C below to perform a fresh execution.
            </div>
          </div>
        ) : (
          <div
            id="no-saved-result-notice"
            style={{
              background: '#f8fafc',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '0.85rem 1rem',
              fontSize: '0.85rem',
              color: '#64748b',
            }}
          >
            No matching completed result currently exists in MySQL for this configuration. Start a new benchmark in Section C below to generate results.
          </div>
        )}
      </div>

      {/* =========================================================================
          SECTION C: Run Benchmark
          ========================================================================= */}
      <div id="section-c-run" style={{ marginBottom: '1.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <span className="badge" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe', fontWeight: 700 }}>C</span>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#0f172a', margin: 0 }}>Run Benchmark</h3>
        </div>

        {/* Action Button & Context Reminder */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
          <button
            id="btn-execute-benchmark"
            onClick={handleExecute}
            disabled={isRunning || isSelectionEmpty || !selectedModelId || !selectedRuntime}
            className="btn btn-primary"
            style={{
              fontWeight: 700,
              padding: '0.65rem 1.5rem',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              opacity: (isRunning || isSelectionEmpty || !selectedModelId || !selectedRuntime) ? 0.6 : 1,
              cursor: (isRunning || isSelectionEmpty || !selectedModelId || !selectedRuntime) ? 'not-allowed' : 'pointer',
            }}
          >
            {isRunning ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Evaluating in Progress...
              </>
            ) : selectedTarget === 'raspberry-pi-4' ? (
              <>
                <Play size={16} fill="currentColor" />
                Profile on Raspberry Pi 4
              </>
            ) : (
              <>
                <Play size={16} fill="currentColor" />
                Run Benchmark
              </>
            )}
          </button>

          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>
            Target: <strong>{currentModel?.display_name || selectedModelId}</strong> • Runtime: <strong>{(selectedRuntime || '').toUpperCase()}</strong> • Device: <strong>{selectedTarget === 'cpu' ? 'Host CPU' : 'Raspberry Pi 4'}</strong>
          </span>
        </div>

        {/* Missing Precomputed Accuracy Pre-Execution Warning (Visible without expanding Advanced Settings) */}
        {isPrecomputedAccuracyMissing && (
          <div
            id="precomputed-accuracy-missing-warning"
            style={{
              marginBottom: '1rem',
              padding: '0.65rem 0.85rem',
              background: '#fffbeb',
              border: '1px solid #fde68a',
              borderRadius: '6px',
              fontSize: '0.8rem',
              color: '#92400e',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <AlertCircle size={15} color="#d97706" style={{ flexShrink: 0 }} />
            <span>
              <strong>Notice: </strong>
              Verified precomputed accuracy is unavailable for {currentModel?.display_name || selectedModelId} on {(selectedRuntime || '').toUpperCase()}. Top-1 and Top-5 accuracy will remain N/A unless fresh live evaluation is performed.
            </span>
          </div>
        )}

        {/* Collapsible Advanced Settings (Optional) */}
        {selectedTarget === 'cpu' && (
          <details
            id="advanced-benchmark-settings"
            style={{
              background: '#f8fafc',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              padding: '0.85rem 1rem',
            }}
          >
            <summary
              style={{
                cursor: 'pointer',
                fontWeight: 600,
                color: '#334155',
                fontSize: '0.875rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
              }}
            >
              <Settings2 size={15} color="#64748b" />
              Advanced Evaluation Settings (Optional)
            </summary>

            <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #e2e8f0' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.35rem' }}>
                    Timed Iterations
                  </label>
                  <input
                    id="input-num-runs"
                    type="number"
                    min="1"
                    max="1000"
                    value={numRuns}
                    disabled={isRunning}
                    onChange={(e) => setNumRuns(e.target.value)}
                    className="saas-input"
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.35rem' }}>
                    Warmup Iterations
                  </label>
                  <input
                    id="input-warmup-runs"
                    type="number"
                    min="0"
                    max="100"
                    value={warmupRuns}
                    disabled={isRunning}
                    onChange={(e) => setWarmupRuns(e.target.value)}
                    className="saas-input"
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569', display: 'block', marginBottom: '0.35rem' }}>
                    Run Session Title (Optional)
                  </label>
                  <input
                    id="input-run-name"
                    type="text"
                    placeholder={`${currentModel?.display_name || selectedModelId}${selectedRuntime ? ` (${selectedRuntime.toUpperCase()})` : ''}`}
                    value={runName}
                    disabled={isRunning}
                    onChange={(e) => setRunName(e.target.value)}
                    className="saas-input"
                  />
                </div>
              </div>

              {/* Metric Checkboxes */}
              <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', alignItems: 'center', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.825rem', fontWeight: 600, color: '#334155' }}>Measure:</span>
                {['latency', 'memory', 'model_size'].map((k) => (
                  <label key={k} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.825rem', color: '#1e293b', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={metrics[k]}
                      disabled={isRunning}
                      onChange={() => toggleMetric(k)}
                      style={{ accentColor: '#2563eb' }}
                    />
                    <span style={{ textTransform: 'capitalize' }}>{k.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>

              {/* Accuracy Evaluation Mode Selection */}
              <div style={{ background: '#ffffff', padding: '0.85rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '0.25rem' }}>
                  <strong>Computational Cost Notice: </strong>
                  Live accuracy evaluation requires evaluating neural networks on real validation images and is computationally expensive. It requires a local ImageNet dataset. Without dataset images, accuracy evaluation will fail. By default, pre-computed verified ImageNet accuracy is attached.
                </div>

                <label htmlFor="radio-accuracy-saved" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: '#0f172a', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    id="radio-accuracy-saved"
                    name="accuracy-eval-mode"
                    checked={!runAccuracyEval}
                    disabled={isRunning}
                    onChange={() => {
                      setRunAccuracyEval(false);
                      setReuseSavedAccuracy(true);
                    }}
                    style={{ accentColor: '#2563eb' }}
                  />
                  <span style={{ fontWeight: 600 }}>Reuse Pre-Computed Verified Accuracy (50,000 ImageNet Samples)</span>
                </label>

                <label htmlFor="radio-accuracy-fresh" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: '#0f172a', cursor: 'pointer' }}>
                  <input
                    type="radio"
                    id="radio-accuracy-fresh"
                    name="accuracy-eval-mode"
                    checked={runAccuracyEval}
                    disabled={isRunning}
                    onChange={() => setIsWarningOpen(true)}
                    style={{ accentColor: '#2563eb' }}
                  />
                  <span>Run Fresh Live Evaluation (ImageNet)</span>
                </label>
              </div>
            </div>
          </details>
        )}
      </div>

      {/* Real-Time Progress Card (Step 4) */}
      {isRunning && progressState && (() => {
        const isAccuracy =
          progressState.stage === 'accuracy_evaluation' ||
          (runAccuracyEval && progressState.stage !== 'performance_benchmarking');
        const stageTitle = isAccuracy
          ? 'Live Accuracy Evaluation (ImageNet Validation)'
          : 'Performance Benchmarking (Latency, Throughput & RAM)';
        const unitLabel = isAccuracy ? 'Images' : 'Iterations';
        const rateLabel = isAccuracy ? 'images/s' : 'iter/s';

        return (
          <div
            id="benchmark-progress-card"
            style={{
              background: '#ffffff',
              border: '2px solid #3b82f6',
              borderRadius: '12px',
              padding: '1.5rem',
              marginBottom: '1.5rem',
              boxShadow: '0 4px 14px rgba(37, 99, 235, 0.12)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Loader2 size={20} className="animate-spin" color="#2563eb" />
                <strong style={{ fontSize: '1rem', color: '#0f172a' }}>Benchmark in Progress</strong>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <span className="badge font-mono" style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe', fontWeight: 600 }}>
                  Stage: {stageTitle}
                </span>
                <span className="badge-running font-mono">
                  Status: {(progressState.status || 'running').toUpperCase()}
                </span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="progress-track" style={{ marginBottom: '1rem', height: '12px' }}>
              <div
                className="progress-fill"
                style={{ width: `${Math.min(100, Math.max(0, progressState.percentage || 0))}%` }}
              />
            </div>

            {/* Live Progress Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', fontSize: '0.875rem' }}>
              <div style={{ background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.2rem' }}>{unitLabel} Completed</div>
                <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.1rem', color: '#0f172a' }}>
                  {(progressState.images_completed || 0).toLocaleString()} / {(progressState.total_images || 0).toLocaleString()}
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.2rem' }}>{unitLabel} Remaining</div>
                <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.1rem', color: '#0f172a' }}>
                  {(progressState.images_remaining || 0).toLocaleString()}
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Percentage Completed</div>
                <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.1rem', color: '#2563eb' }}>
                  {(progressState.percentage || 0).toFixed(1)}%
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Elapsed Time</div>
                <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.1rem', color: '#0f172a' }}>
                  {formatDuration(progressState.elapsed_seconds)}
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Estimated Remaining Time</div>
                <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.1rem', color: '#0f172a' }}>
                  {progressState.estimated_remaining_seconds !== null && progressState.estimated_remaining_seconds !== undefined
                    ? formatDuration(progressState.estimated_remaining_seconds)
                    : 'Calculating...'}
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.75rem 1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ color: '#64748b', fontSize: '0.75rem', marginBottom: '0.2rem' }}>Processing Speed</div>
                <div className="font-mono" style={{ fontWeight: 700, fontSize: '1.1rem', color: '#0f172a' }}>
                  {progressState.processing_speed_fps != null
                    ? `${progressState.processing_speed_fps.toFixed(1)} ${rateLabel}`
                    : 'Calculating...'}
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Completion Banner & Metrics Summary (Step 5) */}
      {completedResult && (() => {
        const mList = completedResult.measurements || [];
        const allFailed = mList.length > 0 && mList.every((m) => m.status === 'failed');
        const anyFailed = mList.some((m) => m.status === 'failed');
        const isPartial = anyFailed || completedResult.evaluation_status === 'partial';
        const isFailed = completedResult.status === 'failed' || allFailed;

        const borderColor = isFailed ? '#fca5a5' : isPartial ? '#fde68a' : '#86efac';

        return (
          <div
            id="benchmark-completion-card"
            style={{
              background: '#ffffff',
              border: `1px solid ${borderColor}`,
              borderRadius: '12px',
              padding: '1.5rem',
              marginBottom: '1.5rem',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {isFailed ? (
                  <>
                    <AlertCircle size={22} color="#dc2626" />
                    <strong style={{ fontSize: '1.05rem', color: '#b91c1c' }}>
                      Evaluation Failed — Run #{completedResult.id}
                    </strong>
                  </>
                ) : isPartial ? (
                  <>
                    <AlertCircle size={22} color="#d97706" />
                    <strong style={{ fontSize: '1.05rem', color: '#b45309' }}>
                      {successMsg || `Evaluation Partially Completed (Stage Failures Disclosed) — Run #${completedResult.id}`}
                    </strong>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={22} color="#16a34a" />
                    <strong style={{ fontSize: '1.05rem', color: '#15803d' }}>
                      {successMsg || `Evaluation Completed — Run #${completedResult.id}`}
                    </strong>
                  </>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {completedResult.provenance === 'edge_impulse_estimation' ? (
                  <span className="badge-estimate">Analytical Estimate</span>
                ) : (
                  <span className="badge-live">Measured Result</span>
                )}
                <span className={`badge ${isFailed ? 'badge-failed' : isPartial ? 'badge-running' : 'badge-completed'}`}>
                  Session: {(completedResult.status || 'completed').toUpperCase()}
                </span>
                {completedResult.evaluation_status && (
                  <span className={`badge ${completedResult.evaluation_status === 'failed' ? 'badge-failed' : completedResult.evaluation_status === 'partial' ? 'badge-running' : 'badge-completed'}`}>
                    Eval: {completedResult.evaluation_status.toUpperCase()}
                  </span>
                )}
                {onNavigate && (
                  <button
                    id="btn-view-run-history"
                    onClick={() => onNavigate('history')}
                    className="btn btn-sm btn-secondary"
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}
                  >
                    <History size={14} />
                    View in Run History
                  </button>
                )}
              </div>
            </div>

            {/* Run Warnings (e.g. Missing Precomputed Accuracy) */}
            {(() => {
              const resultWarnings = Array.isArray(completedResult.warnings)
                ? completedResult.warnings
                : completedResult.warning
                ? [completedResult.warning]
                : [];
              if (resultWarnings.length === 0) return null;
              return (
                <div
                  id="benchmark-result-warning"
                  style={{
                    marginBottom: '1rem',
                    padding: '0.65rem 0.85rem',
                    background: '#fffbeb',
                    border: '1px solid #fde68a',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    color: '#92400e',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                  }}
                >
                  <AlertCircle size={15} color="#d97706" style={{ flexShrink: 0 }} />
                  <div>
                    {resultWarnings.map((w, idx) => (
                      <div key={idx}><strong>Notice: </strong>{w}</div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Measurements Table for this Run */}
            {completedResult.measurements && completedResult.measurements.length > 0 && (
              <div className="saas-table-container">
                <table className="saas-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Runtime</th>
                      <th>Target Device</th>
                      <th>Latency</th>
                      <th>Throughput</th>
                      <th>RAM Footprint</th>
                      <th>Artifact Size</th>
                      <th>Top-1 Accuracy</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {completedResult.measurements.map((m, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600 }}>
                          {m.architecture || m.model_id}
                          {m.status === 'failed' && m.error_message && (
                            <div style={{ fontSize: '0.72rem', color: '#b91c1c', marginTop: '0.15rem' }}>
                              {m.error_message}
                            </div>
                          )}
                        </td>
                        <td>
                          <span className="badge font-mono" style={{ background: '#f1f5f9', color: '#0f172a' }}>
                            {(m.runtime || '').toUpperCase()}
                          </span>
                        </td>
                        <td>
                          <span className="font-mono">{m.device || 'cpu'}</span>
                        </td>
                        <td className="font-mono">
                          {m.latency_mean_ms != null ? `${m.latency_mean_ms} ms` : (m.latency?.mean_ms != null ? `${m.latency.mean_ms} ms` : 'N/A')}
                        </td>
                        <td className="font-mono">
                          {m.throughput_fps != null ? `${m.throughput_fps} fps` : (m.latency?.throughput_fps != null ? `${m.latency.throughput_fps} fps` : 'N/A')}
                        </td>
                        <td className="font-mono">
                          {m.peak_ram_mb != null ? `${m.peak_ram_mb} MiB` : (m.memory?.peak_ram_mb != null ? `${m.memory.peak_ram_mb} MiB` : 'N/A')}
                        </td>
                        <td className="font-mono">
                          {m.file_size_mb != null ? `${m.file_size_mb} MB` : (m.model_size?.file_size_mb != null ? `${m.model_size.file_size_mb} MB` : 'N/A')}
                        </td>
                        <td className="font-mono">
                          {m.top1_accuracy != null ? `${(m.top1_accuracy * 100).toFixed(1)}%` : (m.accuracy?.top1_accuracy != null ? `${(m.accuracy.top1_accuracy * 100).toFixed(1)}%` : 'N/A')}
                        </td>
                        <td>
                          <span className={`badge ${m.status === 'completed' ? 'badge-completed' : 'badge-failed'}`}>
                            {m.status || (completedResult.status === 'failed' ? 'failed' : 'completed')}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })()}

      {/* Failure Message (Step 6) */}
      {errorMsg && (
        <div
          id="benchmark-error-banner"
          style={{
            padding: '1rem 1.25rem',
            borderRadius: '8px',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#b91c1c',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1rem',
            fontSize: '0.875rem',
          }}
        >
          <AlertCircle size={20} style={{ flexShrink: 0 }} />
          <div>
            <strong>Evaluation Alert: </strong> {errorMsg}
          </div>
        </div>
      )}

      {/* Accuracy Warning Modal */}
      <AlertModal
        isOpen={isWarningOpen}
        title="Live Accuracy Benchmarking Warning"
        confirmText="Proceed with Live Evaluation"
        cancelText="Reuse Pre-Computed Accuracy"
        onConfirm={confirmFreshAccuracy}
        onCancel={cancelFreshAccuracy}
      >
        A capped sample threshold of 50 images will be used to limit resource utilization. Running fresh evaluation on thousands of images is computationally intensive. Do you wish to continue?
      </AlertModal>
    </div>
  );
}
