import React from 'react';
import { Layers, Database, Zap, Award, HardDrive } from 'lucide-react';

export default function SummaryCards({ models = [], results = [], totalResults = 0 }) {
  // Lowest Measured Latency evaluates live Host CPU execution only; excludes failed runs and Edge Impulse estimates
  const validLatencies = results.filter((r) => {
    if (r.status === 'failed') return false;
    const meanMs = r.latency?.mean_ms ?? r.latency_mean_ms;
    if (meanMs == null || meanMs <= 0) return false;
    if (r.provenance === 'edge_impulse_estimation') return false;
    return r.device == null || r.device === 'cpu';
  });

  const bestLatency = validLatencies.length
    ? validLatencies.reduce((prev, curr) => {
        const prevMs = prev.latency?.mean_ms ?? prev.latency_mean_ms;
        const currMs = curr.latency?.mean_ms ?? curr.latency_mean_ms;
        return currMs < prevMs ? curr : prev;
      })
    : null;

  const bestLatVal = bestLatency ? (bestLatency.latency?.mean_ms ?? bestLatency.latency_mean_ms) : null;

  // Highest Validated Accuracy: Evaluates verified/validated evaluation accuracy only; excludes failed and analytical estimates
  const validAccuracies = results.filter((r) => {
    if (r.status === 'failed') return false;
    const top1 = r.accuracy?.top1_accuracy ?? r.top1_accuracy;
    if (top1 == null || top1 <= 0) return false;
    if (r.provenance === 'edge_impulse_estimation') return false;
    if (r.status && r.status !== 'completed') return false;
    if (r.accuracy?.accuracy_source && !['live_evaluation', 'precomputed_verified'].includes(r.accuracy.accuracy_source)) return false;
    return true;
  });

  const bestAccuracy = validAccuracies.length
    ? validAccuracies.reduce((prev, curr) => {
        const prevAcc = prev.accuracy?.top1_accuracy ?? prev.top1_accuracy;
        const currAcc = curr.accuracy?.top1_accuracy ?? curr.top1_accuracy;
        return currAcc > prevAcc ? curr : prev;
      })
    : null;

  const bestAccVal = bestAccuracy ? (bestAccuracy.accuracy?.top1_accuracy ?? bestAccuracy.top1_accuracy) : null;

  // Smallest Model Storage Footprint: Verified artifact size data, excluding failed runs
  const validSizes = results.filter((r) => {
    if (r.status === 'failed') return false;
    const sizeMb = r.model_size?.file_size_mb ?? r.file_size_mb;
    return sizeMb != null && sizeMb > 0;
  });

  const bestSize = validSizes.length
    ? validSizes.reduce((prev, curr) => {
        const prevSize = prev.model_size?.file_size_mb ?? prev.file_size_mb;
        const currSize = curr.model_size?.file_size_mb ?? curr.file_size_mb;
        return currSize < prevSize ? curr : prev;
      })
    : null;

  const bestSizeVal = bestSize ? (bestSize.model_size?.file_size_mb ?? bestSize.file_size_mb) : null;

  // Distinct saved runs vs measurements calculation
  const uniqueRunIds = new Set(results.map((r) => r.run_id).filter(Boolean));
  const runCount = uniqueRunIds.size;
  const measurementCount = totalResults || results.length;

  const modelNames = models.length
    ? models.map((m) => m.display_name || m.architecture_name || m.name || m.model_id).join(', ')
    : 'MobileNetV3, ResNet, EfficientNet, DenseNet';

  const cards = [
    {
      id: 'kpi-models',
      label: 'Supported Vision Models',
      value: models.length ? `${models.length} Models` : '4 Models',
      subtext: modelNames,
      icon: Layers,
      iconColor: '#2563eb',
      iconBg: '#eff6ff',
    },
    {
      id: 'kpi-measurements',
      label: 'Saved Runs',
      value: `${runCount} Runs`,
      subtext: `${measurementCount} individual measurement${measurementCount === 1 ? '' : 's'} recorded`,
      icon: Database,
      iconColor: '#0891b2',
      iconBg: '#ecfeff',
    },
    {
      id: 'kpi-latency',
      label: 'Lowest Measured Latency',
      value: bestLatVal != null ? `${bestLatVal.toFixed(2)} ms` : 'N/A',
      subtext: bestLatency
        ? `${bestLatency.architecture || bestLatency.model_id} [${(bestLatency.runtime || '').toUpperCase()}] • Host CPU live execution`
        : 'Host CPU physical measurement only',
      icon: Zap,
      iconColor: '#0284c7',
      iconBg: '#f0f9ff',
    },
    {
      id: 'kpi-accuracy',
      label: 'Highest Validated Accuracy',
      value: bestAccVal != null ? `${(bestAccVal * 100).toFixed(2)}%` : 'N/A',
      subtext: bestAccuracy
        ? `${bestAccuracy.architecture || bestAccuracy.model_id} [${(bestAccuracy.runtime || '').toUpperCase()}] • ${
            bestAccuracy.accuracy?.accuracy_source === 'live_evaluation' ? 'Live ImageNet' : 'Verified Catalog'
          }`
        : 'Verified ImageNet evaluation only',
      icon: Award,
      iconColor: '#059669',
      iconBg: '#ecfdf5',
    },
    {
      id: 'kpi-size',
      label: 'Smallest Model Size',
      value: bestSizeVal != null ? `${parseFloat(bestSizeVal.toFixed(2))} MB` : 'N/A',
      subtext: bestSize
        ? `${bestSize.architecture || bestSize.model_id} [${(bestSize.runtime || '').toUpperCase()}] artifact`
        : 'Verified conversion artifact',
      icon: HardDrive,
      iconColor: '#7c3aed',
      iconBg: '#f5f3ff',
    },
  ];

  return (
    <div id="overview-summary-cards" className="overview-summary-grid">
      {cards.map((card) => {
        const IconComponent = card.icon;
        return (
          <div
            key={card.id}
            id={card.id}
            className="saas-kpi-card"
            style={{
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              padding: '1.15rem 1.25rem',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              transition: 'border-color 150ms ease, box-shadow 150ms ease',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div>
                <span
                  style={{
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    color: '#64748b',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    display: 'block',
                  }}
                >
                  {card.label}
                </span>
                <div
                  style={{
                    fontSize: '1.45rem',
                    fontWeight: 800,
                    marginTop: '0.35rem',
                    color: '#0f172a',
                    letterSpacing: '-0.02em',
                  }}
                >
                  {card.value}
                </div>
              </div>
              <div
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '8px',
                  background: card.iconBg,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                <IconComponent size={18} color={card.iconColor} />
              </div>
            </div>
            <div
              style={{
                fontSize: '0.725rem',
                color: '#64748b',
                marginTop: '0.75rem',
                borderTop: '1px solid #f1f5f9',
                paddingTop: '0.5rem',
              }}
            >
              {card.subtext}
            </div>
          </div>
        );
      })}
    </div>
  );
}
