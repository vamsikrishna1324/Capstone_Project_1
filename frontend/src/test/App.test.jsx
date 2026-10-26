import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';
import * as client from '../api/client';

describe('App Integration - EdgeBench Layout & Overview Dashboard', () => {
  const mockHealth = {
    status: 'ok',
    api: 'online',
    database: { connected: true, host: '127.0.0.1' },
    edge_impulse: { configured: true, status: 'configured' },
  };

  const mockModels = {
    models: [
      {
        model_id: 'mobilenet_v3_small',
        display_name: 'MobileNetV3-Small',
        architecture_name: 'MobileNetV3',
        num_classes: 1000,
        artifacts: {
          onnx: { exists: true, file_size_mb: 9.72 },
          tflite: { exists: true, file_size_mb: 4.9 },
        },
      },
      {
        model_id: 'resnet50',
        display_name: 'ResNet50',
        architecture_name: 'ResNet',
        num_classes: 1000,
        artifacts: {
          onnx: { exists: true, file_size_mb: 102.15 },
          tflite: { exists: true, file_size_mb: 51.11 },
        },
      },
    ],
  };

  const mockResults = {
    total: 3,
    items: [
      // Unified Run #10 containing both MobileNet ONNX and ResNet ONNX
      {
        id: 101,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        provenance: 'live_execution',
        created_at: '2026-10-18T10:00:00Z',
        latency: { mean_ms: 1.48 },
        accuracy: { top1_accuracy: 0.6767 },
      },
      {
        id: 102,
        run_id: 10,
        model_id: 'resnet50',
        architecture: 'ResNet50',
        runtime: 'onnx',
        device: 'cpu',
        provenance: 'live_execution',
        created_at: '2026-10-18T10:00:00Z',
        latency: { mean_ms: 13.02 },
        accuracy: { top1_accuracy: 0.7613 },
      },
      // Distinct Run #12 containing only MobileNet TFLite
      {
        id: 103,
        run_id: 12,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'cpu',
        provenance: 'live_execution',
        created_at: '2026-10-18T12:00:00Z',
        latency: { mean_ms: 2.11 },
        accuracy: { top1_accuracy: 0.6767 },
      },
    ],
  };

  const mockTargets = {
    total_targets: 1,
    targets: [
      {
        target_id: 'raspberry-pi-4',
        display_name: 'Raspberry Pi 4',
        supported_formats: ['tflite'],
        notes: 'Edge Impulse analytical estimate',
      },
    ],
  };

  beforeEach(() => {
    vi.spyOn(client, 'fetchHealth').mockResolvedValue(mockHealth);
    vi.spyOn(client, 'fetchModels').mockResolvedValue(mockModels);
    vi.spyOn(client, 'fetchResults').mockResolvedValue(mockResults);
    vi.spyOn(client, 'fetchEdgeImpulseTargets').mockResolvedValue(mockTargets);
  });

  it('initially loads Overview as a clean summary dashboard without model comparison matrix', async () => {
    render(<App />);

    // Wait for data load
    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    // Overview displays summary dashboard, how to use guide, and quick actions
    expect(screen.getByRole('heading', { level: 1, name: 'Overview' })).toBeInTheDocument();
    expect(screen.getByText('How to Use EdgeBench')).toBeInTheDocument();
    expect(screen.getByText('Platform Workspaces & Navigation')).toBeInTheDocument();
    expect(screen.queryByText('Recent Benchmark Sessions')).not.toBeInTheDocument();

    // Verify comparison charts/tables and matrix controls are strictly absent from Overview
    expect(screen.queryByText('Comparative Benchmark Results')).not.toBeInTheDocument();
    expect(screen.queryByText('Performance Visualizations')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Compare Selected/i })).not.toBeInTheDocument();
  });

  it('renders unified session comparison in Model Comparison tab after user initiates comparison', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    // Navigate to dedicated Model Comparison section
    const compareNavBtn = screen.getByRole('button', { name: /Model Comparison/i });
    fireEvent.click(compareNavBtn);

    await waitFor(() => {
      expect(screen.getByText('MobileNetV3-Small')).toBeInTheDocument();
    });

    // Select ONNX for MobileNetV3-Small and ResNet50
    const mobilenetOnnx = screen.getByLabelText(/ONNX Runtime/i, { selector: '#check-mobilenet_v3_small-onnx' });
    const resnetOnnx = screen.getByLabelText(/ONNX Runtime/i, { selector: '#check-resnet50-onnx' });

    fireEvent.click(mobilenetOnnx);
    fireEvent.click(resnetOnnx);

    // Click "Compare Selected (2)"
    const compareBtn = screen.getByRole('button', { name: /Compare Selected \(2\)/i });
    expect(compareBtn).not.toBeDisabled();
    fireEvent.click(compareBtn);

    // Visualizations and comparison table now appear
    await waitFor(() => {
      expect(screen.getByText('Comparative Benchmark Results')).toBeInTheDocument();
      expect(screen.getByText('Performance Visualizations')).toBeInTheDocument();
    });

    // Verify section order: Comparative Benchmark Results is placed above Performance Visualizations
    const tableHeader = screen.getByText('Comparative Benchmark Results');
    const chartsHeader = screen.getByText('Performance Visualizations');
    expect(tableHeader.compareDocumentPosition(chartsHeader) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    // Unified session preference: Both were in Run #10
    expect(screen.getByText(/Unified Benchmark Session: Run #10/i)).toBeInTheDocument();
    const runBadges = screen.getAllByText('Run #10');
    expect(runBadges.length).toBeGreaterThanOrEqual(2);
  });

  it('detects stale selection when user modifies checkboxes after initiating comparison in Model Comparison', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    // Switch to Model Comparison tab
    fireEvent.click(screen.getByRole('button', { name: /Model Comparison/i }));

    await waitFor(() => {
      expect(screen.getByText('MobileNetV3-Small')).toBeInTheDocument();
    });

    // Select ONNX for MobileNet
    const mobilenetOnnx = screen.getByLabelText(/ONNX Runtime/i, { selector: '#check-mobilenet_v3_small-onnx' });
    fireEvent.click(mobilenetOnnx);

    // Click Compare Selected
    fireEvent.click(screen.getByRole('button', { name: /Compare Selected \(1\)/i }));

    // Modify selection by checking ResNet ONNX
    const resnetOnnx = screen.getByLabelText(/ONNX Runtime/i, { selector: '#check-resnet50-onnx' });
    fireEvent.click(resnetOnnx);

    // Stale notice should now appear
    expect(
      screen.getByText(/Selection changed — comparison results below reflect previous selection/i)
    ).toBeInTheDocument();

    // Clicking Update Comparison refreshes the locked comparison
    const updateBtns = screen.getAllByRole('button', { name: /Update Comparison/i });
    expect(updateBtns.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(updateBtns[0]);

    // Stale banner clears
    expect(
      screen.queryByText(/Selection changed — comparison results below reflect previous selection/i)
    ).not.toBeInTheDocument();
  });

  it('loads supported Edge Impulse targets on startup and passes them to BenchmarkRunner', async () => {
    const targetsSpy = vi.spyOn(client, 'fetchEdgeImpulseTargets');

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    // Switch to benchmark tab
    const benchmarkTabBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(benchmarkTabBtn);

    // Verify BenchmarkRunner has received targets
    await waitFor(() => {
      expect(screen.getAllByText(/Raspberry Pi 4 \(Edge Impulse\)/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Host CPU \(Local Baseline\)/i).length).toBeGreaterThan(0);
    });
  });

  it('resolves model-matrix selections to Host CPU measurements even when a newer Edge Impulse estimate exists', async () => {
    const resultsWithEdge = {
      total: 4,
      items: [
        ...mockResults.items,
        {
          id: 104,
          run_id: 99,
          model_id: 'mobilenet_v3_small',
          architecture: 'MobileNetV3-Small',
          runtime: 'tflite',
          device: 'raspberry-pi-4',
          provenance: 'edge_impulse_estimation',
          created_at: '2026-10-18T14:00:00Z', // Newer than Run #12
          latency: { mean_ms: 21477.0 },
          accuracy: { top1_accuracy: 0.6767 },
        },
      ],
    };
    vi.spyOn(client, 'fetchResults').mockResolvedValue(resultsWithEdge);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    // Switch to Model Comparison tab
    fireEvent.click(screen.getByRole('button', { name: /Model Comparison/i }));

    await waitFor(() => {
      expect(screen.getByText('MobileNetV3-Small')).toBeInTheDocument();
    });

    // Select TFLite for MobileNet in the matrix
    const mobilenetTflite = screen.getByLabelText(/TensorFlow Lite/i, { selector: '#check-mobilenet_v3_small-tflite' });
    fireEvent.click(mobilenetTflite);

    const compareBtn = screen.getByRole('button', { name: /Compare Selected \(1\)/i });
    fireEvent.click(compareBtn);

    await waitFor(() => {
      expect(screen.getByText('Comparative Benchmark Results')).toBeInTheDocument();
    });

    // Must resolve to the Host CPU baseline Run #12 (2.11 ms), NOT the Edge Impulse estimate Run #99
    expect(screen.getAllByText('Run #12').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('2.11 ms').length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText('Run #99')).not.toBeInTheDocument();
    expect(screen.queryByText(/21477/)).not.toBeInTheDocument();
  });

  it('navigates cleanly between tabs using sidebar and quick actions', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    // Check brand in sidebar
    expect(screen.getByText('EdgeBench')).toBeInTheDocument();

    // Check Overview heading
    expect(screen.getByRole('heading', { level: 1, name: 'Overview' })).toBeInTheDocument();

    // Navigate to Run History via sidebar
    const historyTabBtn = screen.getByRole('button', { name: /^Run History$/i });
    fireEvent.click(historyTabBtn);

    await waitFor(() => {
      expect(screen.getByText('Run History & Sessions')).toBeInTheDocument();
    });

    // Navigate back to Overview
    const overviewTabBtn = screen.getByRole('button', { name: /^Overview$/i });
    fireEvent.click(overviewTabBtn);

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'Overview' })).toBeInTheDocument();
    });

    // Navigate via Quick Action button
    const quickCompareBtn = screen.getByRole('button', { name: /Open Comparison Matrix/i });
    fireEvent.click(quickCompareBtn);

    await waitFor(() => {
      expect(screen.getByText('Model Comparison Matrix')).toBeInTheDocument();
    });
  });

  it('toggles sidebar collapse state and persists preference to localStorage', async () => {
    localStorage.clear();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    });

    const sidebar = document.getElementById('app-sidebar');
    expect(sidebar.className).not.toContain('collapsed');

    // Click collapse toggle button
    const collapseBtn = screen.getByRole('button', { name: /Collapse sidebar/i });
    fireEvent.click(collapseBtn);

    expect(sidebar.className).toContain('collapsed');
    expect(localStorage.getItem('edgebench_sidebar_collapsed')).toBe('true');

    // Click expand toggle button
    const expandBtn = screen.getByRole('button', { name: /Expand sidebar/i });
    fireEvent.click(expandBtn);

    expect(sidebar.className).not.toContain('collapsed');
    expect(localStorage.getItem('edgebench_sidebar_collapsed')).toBe('false');
  });
});
