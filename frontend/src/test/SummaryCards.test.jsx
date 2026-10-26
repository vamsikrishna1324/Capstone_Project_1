import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SummaryCards from '../components/SummaryCards';

describe('SummaryCards Component', () => {
  it('renders all KPI summary cards with computed metrics', () => {
    const mockModels = [
      { model_id: 'mobilenet_v3_small' },
      { model_id: 'resnet50' },
      { model_id: 'efficientnet_b0' },
      { model_id: 'densenet121' },
    ];

    const mockResults = [
      {
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        latency: { mean_ms: 1.47 },
        accuracy: { top1_accuracy: 0.6767 },
        model_size: { file_size_mb: 9.72 },
      },
      {
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'pytorch',
        latency: { mean_ms: 13.02 },
        accuracy: { top1_accuracy: 0.8086 },
        model_size: { file_size_mb: 102.15 },
      },
      {
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        latency: { mean_ms: 1.6 },
        accuracy: { top1_accuracy: 0.6764 },
        model_size: { file_size_mb: 4.9 },
      },
    ];

    render(<SummaryCards models={mockModels} results={mockResults} totalResults={3} />);

    // Check models count
    expect(screen.getByText('4 Models')).toBeInTheDocument();

    // Check lowest latency (1.47 ms)
    expect(screen.getByText('1.47 ms')).toBeInTheDocument();

    // Check highest accuracy (80.86%)
    expect(screen.getByText('80.86%')).toBeInTheDocument();

    // Check smallest footprint (4.9 MB)
    expect(screen.getByText('4.9 MB')).toBeInTheDocument();
  });

  it('renders N/A gracefully when metrics are empty', () => {
    render(<SummaryCards models={[]} results={[]} totalResults={0} />);
    expect(screen.getAllByText('N/A').length).toBeGreaterThanOrEqual(3);
  });

  it('strictly ignores Edge Impulse analytical estimates for Lowest Measured Latency', () => {
    const mockResults = [
      {
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        provenance: 'live_execution',
        latency: { mean_ms: 1.47 },
      },
      {
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'raspberry-pi-4',
        provenance: 'edge_impulse_estimation',
        latency: { mean_ms: 0.25 }, // Simulated lower latency
      },
    ];

    render(<SummaryCards models={[{ model_id: 'mobilenet_v3_small' }]} results={mockResults} totalResults={2} />);

    // Lowest Measured Latency must remain the live Host CPU execution (1.47 ms), NOT the 0.25 ms estimate
    expect(screen.getByText('1.47 ms')).toBeInTheDocument();
    expect(screen.queryByText('0.25 ms')).not.toBeInTheDocument();
  });
});
