import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ModelSelector from '../components/ModelSelector';

describe('ModelSelector Component', () => {
  const mockModels = [
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
  ];

  it('renders model cards and accurately displays artifact sizes', () => {
    render(
      <ModelSelector
        models={mockModels}
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onChangeConfigs={() => {}}
      />
    );

    expect(screen.getByText('MobileNetV3-Small')).toBeInTheDocument();
    expect(screen.getByText('ResNet50')).toBeInTheDocument();
    expect(screen.getByText('9.72 MB')).toBeInTheDocument();
    expect(screen.getByText('4.9 MB')).toBeInTheDocument();
  });

  it('handles toggle selection correctly', () => {
    const handleChange = vi.fn();
    render(
      <ModelSelector
        models={mockModels}
        selectedConfigs={{}}
        onChangeConfigs={handleChange}
      />
    );

    const toggleBtn = screen.getByLabelText('Toggle model MobileNetV3-Small');
    fireEvent.click(toggleBtn);

    expect(handleChange).toHaveBeenCalledWith({
      mobilenet_v3_small: ['pytorch', 'onnx', 'tflite'],
    });
  });

  it('handles Compare Selected button state (disabled on empty, clickable when selected)', () => {
    const handleCompare = vi.fn();
    const { rerender } = render(
      <ModelSelector
        models={mockModels}
        selectedConfigs={{}}
        onChangeConfigs={() => {}}
        onInitiateComparison={handleCompare}
      />
    );

    const btnDisabled = screen.getByRole('button', { name: /Compare Selected/i });
    expect(btnDisabled).toBeDisabled();

    // Re-render with active selections
    rerender(
      <ModelSelector
        models={mockModels}
        selectedConfigs={{ mobilenet_v3_small: ['onnx', 'tflite'] }}
        onChangeConfigs={() => {}}
        onInitiateComparison={handleCompare}
      />
    );

    const btnEnabled = screen.getByRole('button', { name: /Compare Selected \(2\)/i });
    expect(btnEnabled).not.toBeDisabled();
    fireEvent.click(btnEnabled);
    expect(handleCompare).toHaveBeenCalledTimes(1);
  });

  it('renders stale warning banner and Update Comparison button when selection changes', () => {
    const handleUpdate = vi.fn();
    render(
      <ModelSelector
        models={mockModels}
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onChangeConfigs={() => {}}
        onInitiateComparison={handleUpdate}
        hasActiveComparison={true}
        isSelectionStale={true}
      />
    );

    expect(
      screen.getByText(/Selection changed — comparison results below reflect previous selection/i)
    ).toBeInTheDocument();

    const updateBtns = screen.getAllByRole('button', { name: /Update Comparison/i });
    expect(updateBtns.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(updateBtns[0]);
    expect(handleUpdate).toHaveBeenCalledTimes(1);
  });

  it('renders individual cards for each saved configuration, displaying separate cards for the same model across different targets/runtimes', () => {
    const savedResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'cpu',
        provenance: 'live_execution',
        status: 'completed',
        latency_mean_ms: 2.11,
        top1_accuracy: 0.6767,
      },
      {
        id: 2,
        run_id: 11,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'tflite',
        device: 'raspberry-pi-4',
        provenance: 'edge_impulse_estimation',
        status: 'completed',
        latency_mean_ms: 21477.0,
      },
    ];

    render(
      <ModelSelector
        models={mockModels}
        savedResults={savedResults}
        selectedConfigs={{}}
        onChangeConfigs={() => {}}
      />
    );

    // Two individual cards rendered for the same model
    expect(screen.getByText('Host CPU')).toBeInTheDocument();
    expect(screen.getByText('Raspberry Pi 4')).toBeInTheDocument();
    expect(screen.getByText('Measured Result')).toBeInTheDocument();
    expect(screen.getByText('Analytical Estimate')).toBeInTheDocument();
    expect(screen.getByText('Session #10')).toBeInTheDocument();
    expect(screen.getByText('Session #11')).toBeInTheDocument();
  });

  it('excludes failed and incomplete benchmark runs from the configuration matrix', () => {
    const savedResults = [
      {
        id: 1,
        run_id: 10,
        model_id: 'mobilenet_v3_small',
        runtime: 'onnx',
        device: 'cpu',
        status: 'completed',
        latency_mean_ms: 1.48,
      },
      {
        id: 2,
        run_id: 11,
        model_id: 'resnet50',
        runtime: 'onnx',
        device: 'cpu',
        status: 'failed',
        latency_mean_ms: null,
      },
      {
        id: 3,
        run_id: 12,
        model_id: 'resnet50',
        runtime: 'tflite',
        device: 'cpu',
        status: 'pending',
        latency_mean_ms: null,
      },
    ];

    render(
      <ModelSelector
        models={mockModels}
        savedResults={savedResults}
        selectedConfigs={{}}
        onChangeConfigs={() => {}}
      />
    );

    expect(screen.getByText('Session #10')).toBeInTheDocument();
    expect(screen.queryByText('Session #11')).not.toBeInTheDocument();
    expect(screen.queryByText('Session #12')).not.toBeInTheDocument();
  });

  it('renders clean empty state with navigation link when no eligible saved benchmark results exist', () => {
    const handleNavigate = vi.fn();
    render(
      <ModelSelector
        models={mockModels}
        savedResults={[]}
        selectedConfigs={{}}
        onChangeConfigs={() => {}}
        onNavigate={handleNavigate}
      />
    );

    expect(
      screen.getByText('No Completed Saved Benchmark Results Available')
    ).toBeInTheDocument();

    const navBtn = screen.getByRole('button', { name: /Go to Run Benchmark/i });
    fireEvent.click(navBtn);
    expect(handleNavigate).toHaveBeenCalledWith('benchmark');
  });
});
