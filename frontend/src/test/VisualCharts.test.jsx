import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import VisualCharts from '../components/VisualCharts';

describe('VisualCharts Component - Performance Visualizations Redesign', () => {
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
      accuracy: { top1_accuracy: 0.6767, top5_accuracy: 0.8741, accuracy_source: 'live_evaluation' },
      model_size: { file_size_mb: 9.72 },
    },
    {
      id: 2,
      run_id: 10,
      model_id: 'resnet50',
      architecture: 'ResNet50',
      runtime: 'onnx',
      device: 'cpu',
      provenance: 'live_execution',
      latency: { mean_ms: 13.02 },
      accuracy: { top1_accuracy: 0.7613, top5_accuracy: 0.9286, accuracy_source: 'live_evaluation' },
      model_size: { file_size_mb: 102.15 },
    },
  ];

  it('renders all three interactive chart types simultaneously in unified section', () => {
    render(<VisualCharts results={mockResults} />);

    // Section title
    expect(screen.getByText('Performance Visualizations')).toBeInTheDocument();

    // Chart 1: Mean Inference Latency
    expect(screen.getByText('Mean Inference Latency')).toBeInTheDocument();
    expect(screen.getByText(/Lower is better ↓/i)).toBeInTheDocument();
    expect(screen.getByText('1.48 ms')).toBeInTheDocument();
    expect(screen.getByText('13.02 ms')).toBeInTheDocument();

    // Chart 2: Model Artifact Size
    expect(screen.getByText('Model Artifact Size')).toBeInTheDocument();
    expect(screen.getByText('9.72 MB')).toBeInTheDocument();
    expect(screen.getByText('102.15 MB')).toBeInTheDocument();
    expect(screen.getByText('Total Size')).toBeInTheDocument();
    expect(screen.getByText('111.9 MB')).toBeInTheDocument(); // 9.72 + 102.15 = 111.87 MB -> 111.9 MB

    // Chart 3: Classification Accuracy
    expect(screen.getByText('Classification Accuracy')).toBeInTheDocument();
    expect(screen.getByText(/Higher is better ↑/i)).toBeInTheDocument();
    expect(screen.getByText('67.67%')).toBeInTheDocument();
    expect(screen.getByText('87.41%')).toBeInTheDocument();
    expect(screen.getByText('76.13%')).toBeInTheDocument();
    expect(screen.getByText('92.86%')).toBeInTheDocument();
  });

  it('supports Linear and Logarithmic scale toggles for Mean Inference Latency', () => {
    const wideRangeResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        provenance: 'live_execution',
        latency: { mean_ms: 1.48 },
        accuracy: { top1_accuracy: 0.6767, top5_accuracy: 0.8741 },
        model_size: { file_size_mb: 9.72 },
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
        accuracy: { top1_accuracy: 0.6764, top5_accuracy: 0.8743 },
        model_size: { file_size_mb: 4.9 },
      },
    ];

    render(<VisualCharts results={wideRangeResults} />);

    // Because ratio > 20x, auto-activates logarithmic scale
    expect(screen.getByText(/Logarithmic Scale \(log10\)/i)).toBeInTheDocument();
    expect(screen.getByText(/~21477 ms \(Est\.\)/i)).toBeInTheDocument();
    expect(screen.getByText(/RPi 4 \(Est\.\)/i)).toBeInTheDocument();

    // Toggle to Linear scale
    const linearBtn = screen.getByRole('button', { name: /^Linear$/i });
    fireEvent.click(linearBtn);

    // Linear scale becomes active
    expect(screen.queryByText(/Logarithmic Scale \(log10\)/i)).not.toBeInTheDocument();

    // Toggle back to Logarithmic
    const logBtn = screen.getByRole('button', { name: /^Logarithmic$/i });
    fireEvent.click(logBtn);
    expect(screen.getByText(/Logarithmic Scale \(log10\)/i)).toBeInTheDocument();
  });

  it('handles missing artifact size by identifying it as unavailable without plotting as zero', () => {
    const resultsWithMissingSize = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        latency: { mean_ms: 1.48 },
        model_size: { file_size_mb: 9.72 },
      },
      {
        id: 2,
        run_id: 10,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'pytorch',
        latency: { mean_ms: 13.02 },
        model_size: { file_size_mb: null }, // In-memory PyTorch weights, no serialized file size
      },
    ];

    render(<VisualCharts results={resultsWithMissingSize} />);

    // Valid size is displayed
    expect(screen.getByText('9.72 MB')).toBeInTheDocument();

    // Missing size is identified as Unavailable in legend (never 0 MB)
    expect(screen.getByText('Unavailable')).toBeInTheDocument();
    expect(screen.queryByText('0 MB')).not.toBeInTheDocument();

    // Notice that total is partial / excludes unavailable
    expect(screen.getByText(/Total excludes configurations without verified physical artifacts/i)).toBeInTheDocument();
    expect(screen.getByText('1/2 Available')).toBeInTheDocument();
  });

  it('handles missing accuracy values as unavailable without plotting as zero percent', () => {
    const resultsWithMissingAccuracy = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        accuracy: { top1_accuracy: 0.6767, top5_accuracy: null }, // Missing top-5
      },
      {
        id: 2,
        run_id: 10,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'pytorch',
        accuracy: null, // Entire accuracy unmeasured
      },
    ];

    render(<VisualCharts results={resultsWithMissingAccuracy} />);

    // Top-1 for MobileNet is shown
    expect(screen.getByText('67.67%')).toBeInTheDocument();

    // Missing Top-5 is identified as unavailable
    expect(screen.getAllByText(/Top-5 accuracy unavailable \(N\/A\)/i).length).toBeGreaterThanOrEqual(1);

    // ResNet missing Top-1 and Top-5 are identified as unavailable
    expect(screen.getByText(/Top-1 accuracy unavailable \(N\/A\)/i)).toBeInTheDocument();

    // Neither is plotted as 0.00%
    expect(screen.queryByText('0.00%')).not.toBeInTheDocument();
  });

  it('allows switching views to isolate specific chart types', () => {
    render(<VisualCharts results={mockResults} />);

    // Initially all 3 are visible
    expect(screen.getByText('Mean Inference Latency')).toBeInTheDocument();
    expect(screen.getByText('Model Artifact Size')).toBeInTheDocument();
    expect(screen.getByText('Classification Accuracy')).toBeInTheDocument();

    // Switch to Latency view
    const latencyViewBtn = screen.getByRole('button', { name: /Mean Latency/i });
    fireEvent.click(latencyViewBtn);

    expect(screen.getByText('Mean Inference Latency')).toBeInTheDocument();
    expect(screen.queryByText('Model Artifact Size')).not.toBeInTheDocument();
    expect(screen.queryByText('Classification Accuracy')).not.toBeInTheDocument();

    // Switch to Artifact Size view
    const sizeViewBtn = screen.getByRole('button', { name: /Artifact Size/i });
    fireEvent.click(sizeViewBtn);

    expect(screen.getByText('Model Artifact Size')).toBeInTheDocument();
    expect(screen.queryByText('Mean Inference Latency')).not.toBeInTheDocument();
    expect(screen.queryByText('Classification Accuracy')).not.toBeInTheDocument();

    // Switch to Accuracy view
    const accViewBtn = screen.getByRole('button', { name: /^Accuracy$/i });
    fireEvent.click(accViewBtn);

    expect(screen.getByText('Classification Accuracy')).toBeInTheDocument();
    expect(screen.queryByText('Mean Inference Latency')).not.toBeInTheDocument();
    expect(screen.queryByText('Model Artifact Size')).not.toBeInTheDocument();

    // Return to All
    const allViewBtn = screen.getByRole('button', { name: /All Visualizations/i });
    fireEvent.click(allViewBtn);

    expect(screen.getByText('Mean Inference Latency')).toBeInTheDocument();
    expect(screen.getByText('Model Artifact Size')).toBeInTheDocument();
    expect(screen.getByText('Classification Accuracy')).toBeInTheDocument();
  });

  it('renders clean empty state when results prop is empty', () => {
    render(<VisualCharts results={[]} />);

    expect(screen.getByText('No Benchmark Results Selected')).toBeInTheDocument();
    expect(
      screen.getByText(/Select model configurations in the Saved Results Selection Matrix above/i)
    ).toBeInTheDocument();
  });
});
