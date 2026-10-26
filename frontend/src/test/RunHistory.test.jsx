import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import RunHistory from '../components/RunHistory';
import * as client from '../api/client';

describe('RunHistory Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders clean empty state when no benchmark sessions exist', () => {
    render(<RunHistory results={[]} />);

    // Empty state card
    expect(screen.getByText('No Benchmark Sessions Recorded Yet')).toBeInTheDocument();
    expect(
      screen.getByText(/Execute a benchmark run from the/i)
    ).toBeInTheDocument();
    expect(screen.getByText('0 Saved Sessions')).toBeInTheDocument();

    // Table headers should NOT be rendered when empty
    expect(screen.queryByText('Models Included')).not.toBeInTheDocument();
  });

  it('renders sessions list correctly when historical results exist', () => {
    const mockResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        status: 'completed',
        created_at: '2026-10-18T10:00:00Z',
      },
      {
        id: 2,
        run_id: 10,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'onnx',
        device: 'cpu',
        status: 'completed',
        created_at: '2026-10-18T10:00:00Z',
      },
      {
        id: 3,
        run_id: 12,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'cpu',
        status: 'completed',
        created_at: '2026-10-18T12:00:00Z',
      },
    ];

    render(<RunHistory results={mockResults} />);

    expect(screen.getByText('2 Saved Sessions')).toBeInTheDocument();
    expect(screen.getByText('#10')).toBeInTheDocument();
    expect(screen.getByText('#12')).toBeInTheDocument();
    expect(screen.getByText('mobilenet_v3_small, resnet50')).toBeInTheDocument();
  });

  it('handles inspect drill-down and back navigation', async () => {
    const mockResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        status: 'completed',
        created_at: '2026-10-18T10:00:00Z',
      },
    ];

    const mockDetail = {
      id: 10,
      run_name: 'Edge Comparative Session',
      device: 'cpu',
      status: 'completed',
      provenance: 'benchmark_execution',
      created_at: '2026-10-18T10:00:00Z',
      measurements: [
        {
          id: 1,
          run_id: 10,
          model_id: 'mobilenet_v3_small',
          architecture: 'MobileNetV3-Small',
          runtime: 'onnx',
          accuracy: { top1_accuracy: 0.6767 },
          latency: { mean_ms: 1.48 },
          memory: { peak_ram_mb: 652.6 },
          model_size: { file_size_mb: 9.72 },
          status: 'completed',
        },
      ],
    };

    const inspectSpy = vi.spyOn(client, 'fetchBenchmarkRun').mockResolvedValue(mockDetail);

    render(<RunHistory results={mockResults} />);

    // Click inspect button for run 10
    const inspectBtn = screen.getByRole('button', { name: /Inspect/i });
    fireEvent.click(inspectBtn);

    expect(inspectSpy).toHaveBeenCalledWith(10);

    // Detailed run inspection view should appear
    await waitFor(() => {
      expect(screen.getByText('Edge Comparative Session')).toBeInTheDocument();
    });

    expect(screen.getByText('Run #10')).toBeInTheDocument();
    expect(screen.getByText('Associated Measurements (1)')).toBeInTheDocument();
    expect(screen.getByText('1.48 ms')).toBeInTheDocument();
    expect(screen.getByText('67.67%')).toBeInTheDocument();

    // Click Back to return to sessions list
    const backBtn = screen.getByRole('button', { name: /Back to Sessions List/i });
    fireEvent.click(backBtn);

    expect(screen.queryByText('Edge Comparative Session')).not.toBeInTheDocument();
    expect(screen.getByText('#10')).toBeInTheDocument();
  });

  it('distinguishes job completion from evaluation failure and preserves valid latency metrics', async () => {
    const mockResults = [
      {
        id: 4,
        run_id: 4,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'onnx',
        device: 'cpu',
        status: 'failed',
        created_at: '2026-10-24T06:00:00Z',
      },
    ];

    const mockDetail = {
      id: 4,
      run_name: 'ResNet50 Live Run',
      device: 'cpu',
      status: 'completed', // Job orchestration completed
      evaluation_status: 'failed', // Evaluation measurement failed
      provenance: 'live_execution',
      created_at: '2026-10-24T06:00:00Z',
      measurements: [
        {
          id: 4,
          run_id: 4,
          model_id: 'resnet50',
          architecture: 'ResNet50',
          runtime: 'onnx',
          accuracy: { top1_accuracy: null },
          latency: { mean_ms: 11.46, throughput_fps: 87.2 },
          memory: { peak_ram_mb: 208.9 },
          model_size: { file_size_mb: 97.4 },
          status: 'failed',
          error_message: 'Accuracy benchmarking failed: No evaluation dataset provided at data/imagenet/val',
        },
      ],
    };

    vi.spyOn(client, 'fetchBenchmarkRun').mockResolvedValue(mockDetail);

    render(<RunHistory results={mockResults} />);

    // Sessions table should display Failed evaluation status
    expect(screen.getByText('Failed')).toBeInTheDocument();

    const inspectBtn = screen.getByRole('button', { name: /Inspect/i });
    fireEvent.click(inspectBtn);

    await waitFor(() => {
      expect(screen.getByText('ResNet50 Live Run')).toBeInTheDocument();
    });

    // Both statuses are represented separately
    expect(screen.getByText('Job: COMPLETED')).toBeInTheDocument();
    expect(screen.getByText('Evaluation: FAILED')).toBeInTheDocument();

    // Diagnostic banner appears
    expect(screen.getByText(/Evaluation Outcome Notice/i)).toBeInTheDocument();
    expect(
      screen.getAllByText(/Accuracy benchmarking failed: No evaluation dataset provided/i).length
    ).toBeGreaterThan(0);

    // Valid latency is preserved, accuracy is N/A
    expect(screen.getByText('11.46 ms')).toBeInTheDocument();
    expect(screen.getByText('87.2 fps')).toBeInTheDocument();
    expect(screen.getByText('N/A')).toBeInTheDocument();
  });
});
