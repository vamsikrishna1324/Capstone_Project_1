import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import ComparisonTable from '../components/ComparisonTable';

describe('ComparisonTable Component - Updated Columns & Single Notice', () => {
  it('renders approved 9 table columns in order and excludes P95 Latency and Parameters', () => {
    const mockResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        latency: { mean_ms: 1.48, p95_ms: 1.73, throughput_fps: 681.3 },
        accuracy: { top1_accuracy: 0.6767, top5_accuracy: 0.8741 },
        memory: { peak_ram_mb: 652.6 },
        model_size: { file_size_mb: 9.72, parameter_count: 2542856 },
      },
    ];

    render(<ComparisonTable results={mockResults} />);

    // Check table headers in order
    const headers = screen.getAllByRole('columnheader').map((th) => th.textContent.trim());
    expect(headers).toEqual([
      'Model / Architecture',
      'Runtime',
      'Device',
      'Source Run',
      'Top-1 Accuracy',
      'Top-5 Accuracy',
      'Mean Latency',
      'Throughput',
      'Peak RAM',
    ]);

    // Explicitly verify removed columns are absent
    expect(screen.queryByText('P95 Latency')).not.toBeInTheDocument();
    expect(screen.queryByText('Parameters')).not.toBeInTheDocument();
    expect(screen.queryByText('Model Size')).not.toBeInTheDocument();
  });

  it('correctly maps target devices and displays N/A when device is missing', () => {
    const mockResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        latency: { mean_ms: 1.48 },
        accuracy: { top1_accuracy: 0.6767 },
      },
      {
        id: 2,
        run_id: 11,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'raspberry-pi-4',
        provenance: 'edge_impulse_estimation',
        latency: { mean_ms: 21477.0 },
      },
      {
        id: 3,
        run_id: 12,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'pytorch',
        device: null, // Device missing
        latency: { mean_ms: 13.02 },
      },
    ];

    render(<ComparisonTable results={mockResults} />);

    // Device column readable labels
    expect(screen.getByText('Host CPU')).toBeInTheDocument();
    expect(screen.getByText('Raspberry Pi 4')).toBeInTheDocument();

    // Missing device displayed as N/A without fabrication
    const naBadges = screen.getAllByText('N/A');
    expect(naBadges.length).toBeGreaterThanOrEqual(1);
  });

  it('excludes Host Machine Simulation Notice and Multi-Session Comparison notice', () => {
    const mockResults = [
      {
        id: 1,
        run_id: 17,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        latency: { mean_ms: 1.48 },
      },
      {
        id: 2,
        run_id: 21,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'onnx',
        device: 'cpu',
        latency: { mean_ms: 24.5 },
      },
    ];

    render(
      <ComparisonTable
        results={mockResults}
        sessionInfo={{ isSingleSession: false, sourceRuns: [21, 17] }}
      />
    );

    // Both notices must be absent
    expect(screen.queryByText(/Host Machine Simulation Notice/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Multi-Session Comparison:/i)).not.toBeInTheDocument();
  });

  it('renders only the Edge Impulse Analytical Estimation Notice when edge impulse results exist', () => {
    const mockResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        provenance: 'live_execution',
        latency: { mean_ms: 1.48 },
      },
      {
        id: 2,
        run_id: 55,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'raspberry-pi-4',
        provenance: 'edge_impulse_estimation',
        latency: { mean_ms: 21477.0 },
      },
    ];

    render(<ComparisonTable results={mockResults} />);

    // Edge Impulse Analytical Estimation Notice is present
    expect(screen.getByText(/Edge Impulse Analytical Estimation Notice:/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Latency is an analytical estimate, not a physical-device measurement/i)
    ).toBeInTheDocument();

    // Verified that other notices remain strictly absent
    expect(screen.queryByText(/Host Machine Simulation Notice/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Multi-Session Comparison:/i)).not.toBeInTheDocument();

    // Source run and row provenance markers are preserved
    expect(screen.getByText('Run #10')).toBeInTheDocument();
    expect(screen.getByText('Run #55')).toBeInTheDocument();
    expect(screen.getByText(/~21477\.00 ms/i)).toBeInTheDocument();
    expect(screen.getByText('Est.')).toBeInTheDocument();
  });

  it('renders clean empty state when results are empty', () => {
    render(<ComparisonTable results={[]} />);
    expect(
      screen.getByText('Select models and runtimes to compare their performance.')
    ).toBeInTheDocument();
  });
});
