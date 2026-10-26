import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import BenchmarkRunner from '../components/BenchmarkRunner';
import * as client from '../api/client';

describe('BenchmarkRunner Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('disables run button when no configurations are selected', () => {
    render(<BenchmarkRunner selectedConfigs={{}} onBenchmarkSuccess={() => {}} />);
    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    expect(runBtn).toBeDisabled();
  });

  it('triggers warning modal when user selects fresh live accuracy evaluation', () => {
    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    const liveAccuracyRadio = screen.getByLabelText(/Run Fresh Live Evaluation/i);
    fireEvent.click(liveAccuracyRadio);

    // Warning dialog should appear
    expect(screen.getByText(/Live Accuracy Benchmarking Warning/i)).toBeInTheDocument();
    expect(screen.getByText(/A capped sample threshold/i)).toBeInTheDocument();
  });

  it('preserves unchanged Host CPU routing to runBenchmarkComparison', async () => {
    const mockRunResult = { id: 42, measurements: [] };
    const runSpy = vi.spyOn(client, 'runBenchmarkComparison').mockResolvedValue(mockRunResult);
    const profileSpy = vi.spyOn(client, 'profileModelEdgeImpulse');

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    // Host CPU is default
    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(runSpy).toHaveBeenCalledTimes(1);
    });

    expect(runSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        device: 'cpu',
        model_ids: ['mobilenet_v3_small'],
        runtimes: ['onnx'],
      })
    );
    expect(profileSpy).not.toHaveBeenCalled();
    expect(screen.getByText(/Benchmark completed successfully! Run ID #42/i)).toBeInTheDocument();
  });

  it('routes requests exclusively to profileModelEdgeImpulse when Raspberry Pi 4 is selected', async () => {
    const mockProfileResult = {
      id: 99,
      status: 'success',
      model_id: 'mobilenet_v3_small',
      runtime: 'tflite',
      latency_ms: 21477.0,
      provenance: 'edge_impulse_estimation',
      measurements: [{ model_id: 'mobilenet_v3_small', runtime: 'tflite' }],
    };
    const profileSpy = vi.spyOn(client, 'profileModelEdgeImpulse').mockResolvedValue(mockProfileResult);
    const runSpy = vi.spyOn(client, 'runBenchmarkComparison');
    const successCallback = vi.fn();

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['tflite'] }}
        targets={[
          {
            target_id: 'raspberry-pi-4',
            display_name: 'Raspberry Pi 4',
            supported_formats: ['tflite'],
          },
        ]}
        onBenchmarkSuccess={successCallback}
      />
    );

    // Select Raspberry Pi 4
    const rpi4Radio = screen.getByLabelText(/Raspberry Pi 4/i);
    fireEvent.click(rpi4Radio);

    // Button label reflects Raspberry Pi 4
    const profileBtn = screen.getByRole('button', { name: /Profile on Raspberry Pi 4/i });
    fireEvent.click(profileBtn);

    await waitFor(() => {
      expect(profileSpy).toHaveBeenCalledTimes(1);
    });

    expect(profileSpy).toHaveBeenCalledWith({
      model_id: 'mobilenet_v3_small',
      target_device: 'raspberry-pi-4',
      runtime: 'tflite',
      batch_size: 1,
      reuse_completed: true,
      reuse_saved_accuracy: true,
    });
    // Never calls the local benchmark endpoint
    expect(runSpy).not.toHaveBeenCalled();
    expect(successCallback).toHaveBeenCalledWith(mockProfileResult);
    expect(screen.getByText(/Edge Impulse profiling completed successfully! Run ID #99/i)).toBeInTheDocument();
  });

  it('rejects ONNX runtime and prevents submission when Raspberry Pi 4 is selected', async () => {
    const profileSpy = vi.spyOn(client, 'profileModelEdgeImpulse');
    const runSpy = vi.spyOn(client, 'runBenchmarkComparison');

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    // Select Raspberry Pi 4
    const rpi4Radio = screen.getByLabelText(/Raspberry Pi 4/i);
    fireEvent.click(rpi4Radio);

    const profileBtn = screen.getByRole('button', { name: /Profile on Raspberry Pi 4/i });
    fireEvent.click(profileBtn);

    // Should immediately display validation error and block call
    expect(
      screen.getByText(/Edge Impulse profiling on Raspberry Pi 4 supports TFLite only/i)
    ).toBeInTheDocument();
    expect(profileSpy).not.toHaveBeenCalled();
    expect(runSpy).not.toHaveBeenCalled();
  });

  it('handles Edge Impulse profiling errors gracefully without crashing', async () => {
    vi.spyOn(client, 'profileModelEdgeImpulse').mockRejectedValue(
      new Error('Edge Impulse target authentication failed')
    );

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['tflite'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    const rpi4Radio = screen.getByLabelText(/Raspberry Pi 4/i);
    fireEvent.click(rpi4Radio);

    const profileBtn = screen.getByRole('button', { name: /Profile on Raspberry Pi 4/i });
    fireEvent.click(profileBtn);

    await waitFor(() => {
      expect(
        screen.getByText('Edge Impulse target authentication failed')
      ).toBeInTheDocument();
    });

    // Button should be re-enabled after failure
    expect(profileBtn).not.toBeDisabled();
  });

  it('renders structured sections A, B, and C with summary bar and collapsible advanced settings', () => {
    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    // Section headers
    expect(screen.getByText('Select Configuration')).toBeInTheDocument();
    expect(screen.getByText('Existing Result Check')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Run Benchmark' })).toBeInTheDocument();

    // Summary bar in Section A
    expect(screen.getByText('Selected Configuration Summary')).toBeInTheDocument();
    expect(screen.getByText(/ONNX on Host CPU \(Local Baseline\)/i)).toBeInTheDocument();

    // Advanced settings collapsed
    expect(screen.getByText(/Advanced Evaluation Settings \(Optional\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Hardware Disclaimer/i)).toBeInTheDocument();
  });

  it('displays matching completed saved result and executes Use Saved Result without progress polling', () => {
    const mockSuccess = vi.fn();
    const savedResults = [
      {
        id: 101,
        run_id: 12,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        status: 'completed',
        provenance: 'live_execution',
        latency_mean_ms: 1.48,
        throughput_fps: 675.2,
        top1_accuracy: 0.6767,
        peak_ram_mb: 450.2,
        file_size_mb: 9.72,
      },
    ];

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        savedResults={savedResults}
        onBenchmarkSuccess={mockSuccess}
      />
    );

    // Existing result card should appear in Section B
    expect(screen.getByText(/Matching Completed Result Available \(Run ID #12\)/i)).toBeInTheDocument();
    expect(screen.getByText('1.48 ms')).toBeInTheDocument();
    expect(screen.getByText('67.7%')).toBeInTheDocument();

    // Click Use Saved Result
    const useBtn = screen.getByRole('button', { name: /Use Saved Result/i });
    fireEvent.click(useBtn);

    expect(mockSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 12,
        measurements: [savedResults[0]],
      })
    );
    expect(screen.getByText(/Existing saved benchmark result reused! Run ID #12 loaded/i)).toBeInTheDocument();
    // Progress card should NOT be rendered for a reused result
    expect(screen.queryByText(/Images Completed/i)).not.toBeInTheDocument();
  });

  it('strictly excludes failed measurements from result reuse in Section B', () => {
    const savedResults = [
      {
        id: 4,
        run_id: 4,
        model_id: 'mobilenet_v3_small',
        architecture: 'MobileNetV3-Small',
        runtime: 'onnx',
        device: 'cpu',
        status: 'failed', // Failed measurement must never be reused
        provenance: 'live_execution',
        latency_mean_ms: 1.48,
        throughput_fps: 675.2,
        top1_accuracy: null,
      },
    ];

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        savedResults={savedResults}
        onBenchmarkSuccess={() => {}}
      />
    );

    // Reusable result card should NOT appear
    expect(screen.queryByText(/Matching Completed Result Available/i)).not.toBeInTheDocument();
    expect(screen.getByText(/No matching completed result currently exists in MySQL/i)).toBeInTheDocument();
    expect(screen.getByText(/Existing Result Ineligible for Reuse \(Run #4\)/i)).toBeInTheDocument();
  });

  it('initiates live polling when backend returns running status and renders performance benchmarking progress', async () => {
    const mockInitialRun = { id: 77, status: 'running', measurements: [] };
    const mockProgress = {
      status: 'running',
      stage: 'performance_benchmarking',
      images_completed: 40,
      total_images: 110,
      images_remaining: 70,
      percentage: 36.4,
      elapsed_seconds: 1.5,
      estimated_remaining_seconds: 2.5,
      processing_speed_fps: 26.7,
    };
    const mockCompletedProgress = {
      status: 'completed',
      stage: 'performance_benchmarking',
      images_completed: 110,
      total_images: 110,
      images_remaining: 0,
      percentage: 100.0,
      elapsed_seconds: 4.0,
      estimated_remaining_seconds: 0.0,
      processing_speed_fps: 27.5,
    };
    const mockFinalRun = {
      id: 77,
      status: 'completed',
      evaluation_status: 'completed',
      provenance: 'live_execution',
      measurements: [
        {
          id: 771,
          run_id: 77,
          model_id: 'mobilenet_v3_small',
          architecture: 'MobileNetV3-Small',
          runtime: 'onnx',
          device: 'cpu',
          status: 'completed',
          latency_mean_ms: 1.55,
          throughput_fps: 645.1,
          peak_ram_mb: 450.0,
          file_size_mb: 9.7,
          top1_accuracy: 0.6767,
        },
      ],
    };

    vi.spyOn(client, 'runBenchmarkComparison').mockResolvedValue(mockInitialRun);
    const progressSpy = vi
      .spyOn(client, 'fetchBenchmarkProgress')
      .mockResolvedValueOnce(mockProgress)
      .mockResolvedValue(mockCompletedProgress);
    const runSpy = vi.spyOn(client, 'fetchBenchmarkRun').mockResolvedValue(mockFinalRun);
    const successCb = vi.fn();

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onBenchmarkSuccess={successCb}
      />
    );

    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(runBtn);

    // Should enter running state and start polling
    await waitFor(() => {
      expect(progressSpy).toHaveBeenCalledWith(77);
    });

    // Should show real progress bar with performance stage and iteration units (not images)
    expect(screen.getByText(/Stage: Performance Benchmarking/i)).toBeInTheDocument();
    expect(screen.getByText('Iterations Completed')).toBeInTheDocument();
    expect(screen.getByText('40 / 110')).toBeInTheDocument();
    expect(screen.getByText('36.4%')).toBeInTheDocument();

    // After completion poll resolves
    await waitFor(() => {
      expect(screen.getByText(/Benchmark completed successfully! Run ID #77/i)).toBeInTheDocument();
    });

    expect(successCb).toHaveBeenCalledWith(mockFinalRun);
    expect(screen.getByText('1.55 ms')).toBeInTheDocument();
  });

  it('clears stale success and completion banners when user changes configuration selection', async () => {
    const mockRunResult = {
      id: 88,
      status: 'completed',
      measurements: [
        {
          id: 881,
          run_id: 88,
          model_id: 'mobilenet_v3_small',
          architecture: 'MobileNetV3-Small',
          runtime: 'onnx',
          device: 'cpu',
          status: 'completed',
          latency_mean_ms: 1.5,
        },
      ],
    };
    vi.spyOn(client, 'runBenchmarkComparison').mockResolvedValue(mockRunResult);

    render(
      <BenchmarkRunner
        models={[
          { model_id: 'mobilenet_v3_small', display_name: 'MobileNetV3-Small' },
          { model_id: 'resnet50', display_name: 'ResNet-50' },
        ]}
        onBenchmarkSuccess={() => {}}
      />
    );

    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(screen.getByText(/Benchmark completed successfully! Run ID #88/i)).toBeInTheDocument();
    });

    // Now switch format to TensorFlow Lite
    const tfliteBtn = screen.getByRole('button', { name: /TensorFlow Lite/i });
    fireEvent.click(tfliteBtn);

    // Stale success banner from Run #88 must be cleared
    await waitFor(() => {
      expect(screen.queryByText(/Benchmark completed successfully! Run ID #88/i)).not.toBeInTheDocument();
    });
  });

  it('discloses partial outcomes in completion card when some measurements fail', async () => {
    const mockPartialRun = {
      id: 99,
      status: 'completed',
      evaluation_status: 'partial',
      provenance: 'live_execution',
      measurements: [
        {
          id: 991,
          run_id: 99,
          model_id: 'mobilenet_v3_small',
          architecture: 'MobileNetV3-Small',
          runtime: 'onnx',
          device: 'cpu',
          status: 'completed',
          latency_mean_ms: 1.48,
          throughput_fps: 675.2,
          peak_ram_mb: 450.0,
          file_size_mb: 9.7,
        },
        {
          id: 992,
          run_id: 99,
          model_id: 'mobilenet_v3_small',
          architecture: 'MobileNetV3-Small',
          runtime: 'tflite',
          device: 'cpu',
          status: 'failed',
          error_message: 'Accuracy evaluation dataset not mounted.',
        },
      ],
    };
    vi.spyOn(client, 'runBenchmarkComparison').mockResolvedValue(mockPartialRun);

    render(
      <BenchmarkRunner
        selectedConfigs={{ mobilenet_v3_small: ['onnx'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/Benchmark completed with partial results\. Run ID #99 recorded\./i)
      ).toBeInTheDocument();
    });

    // Discloses failed stage and measurement error message while preserving latency
    expect(screen.getByText(/Accuracy evaluation dataset not mounted/i)).toBeInTheDocument();
    expect(screen.getByText('1.48 ms')).toBeInTheDocument();
    expect(screen.getByText('Eval: PARTIAL')).toBeInTheDocument();
  });

  it('displays pre-execution warning when precomputed accuracy is unavailable for DenseNet121 TFLite', () => {
    render(
      <BenchmarkRunner
        selectedConfigs={{ densenet121: ['tflite'] }}
        models={[
          {
            model_id: 'densenet121',
            display_name: 'DenseNet121',
            artifacts: {
              tflite: { format: 'tflite', exists: true, has_precomputed_accuracy: false },
            },
          },
        ]}
        onBenchmarkSuccess={() => {}}
      />
    );

    // Should display the precomputed accuracy missing notice
    expect(screen.getByText(/Verified precomputed accuracy is unavailable for DenseNet121 on TFLITE/i)).toBeInTheDocument();
  });

  it('displays returned backend warning notice in completed result state', async () => {
    const mockRun = {
      id: 8,
      status: 'completed',
      evaluation_status: 'completed',
      provenance: 'live_execution',
      warnings: [
        'Verified precomputed accuracy is unavailable for DenseNet121 on TFLITE. Top-1 and Top-5 accuracy remain N/A unless fresh live evaluation is performed.',
      ],
      measurements: [
        {
          id: 8,
          run_id: 8,
          model_id: 'densenet121',
          architecture: 'DenseNet121',
          runtime: 'tflite',
          device: 'cpu',
          status: 'completed',
          latency_mean_ms: 101.55,
          throughput_fps: 9.85,
          peak_ram_mb: 472.78,
          file_size_mb: 15.35,
          top1_accuracy: null,
        },
      ],
    };
    vi.spyOn(client, 'runBenchmarkComparison').mockResolvedValue(mockRun);

    render(
      <BenchmarkRunner
        selectedConfigs={{ densenet121: ['tflite'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(runBtn);

    await waitFor(() => {
      expect(screen.getByText(/Benchmark completed successfully! Run ID #8/i)).toBeInTheDocument();
    });

    const resultWarning = document.getElementById('benchmark-result-warning');
    expect(resultWarning).toBeInTheDocument();
    expect(resultWarning).toHaveTextContent(/Verified precomputed accuracy is unavailable for DenseNet121 on TFLITE/i);
    expect(screen.getByText('101.55 ms')).toBeInTheDocument();
    expect(screen.getByText('N/A')).toBeInTheDocument();
  });

  it('displays pre-execution warning visibly even when Advanced Settings accordion is collapsed', () => {
    render(
      <BenchmarkRunner
        selectedConfigs={{ densenet121: ['tflite'] }}
        models={[
          {
            model_id: 'densenet121',
            display_name: 'DenseNet121',
            artifacts: {
              tflite: { format: 'tflite', exists: true, has_precomputed_accuracy: false },
            },
          },
        ]}
        onBenchmarkSuccess={() => {}}
      />
    );

    const advancedDetails = document.getElementById('advanced-benchmark-settings');
    expect(advancedDetails).toBeInTheDocument();
    // Default HTML details accordion state is collapsed (not open)
    expect(advancedDetails.open).toBe(false);

    // Warning is outside the collapsed details and directly in document
    const warningEl = document.getElementById('precomputed-accuracy-missing-warning');
    expect(warningEl).toBeInTheDocument();
    expect(advancedDetails.contains(warningEl)).toBe(false);
    expect(warningEl).toHaveTextContent(/Verified precomputed accuracy is unavailable for DenseNet121 on TFLITE/i);
  });

  it('includes explanatory warning when reusing a saved result with missing accuracy', () => {
    const mockReusedItem = {
      id: 8,
      run_id: 8,
      model_id: 'densenet121',
      architecture: 'DenseNet121',
      runtime: 'tflite',
      device: 'cpu',
      status: 'completed',
      latency_mean_ms: 101.55,
      throughput_fps: 9.85,
      peak_ram_mb: 472.78,
      file_size_mb: 15.35,
      top1_accuracy: null,
      accuracy: null,
      provenance: 'live_execution',
    };

    const successCb = vi.fn();

    render(
      <BenchmarkRunner
        selectedConfigs={{ densenet121: ['tflite'] }}
        savedResults={[mockReusedItem]}
        onBenchmarkSuccess={successCb}
      />
    );

    // Step 3 shows reuse button
    const reuseBtn = screen.getByRole('button', { name: /Use Saved Result/i });
    expect(reuseBtn).toBeInTheDocument();
    fireEvent.click(reuseBtn);

    // Completed result card renders with warning
    const resultWarning = document.getElementById('benchmark-result-warning');
    expect(resultWarning).toBeInTheDocument();
    expect(resultWarning).toHaveTextContent(/Verified precomputed accuracy is unavailable for DenseNet121 on TFLITE/i);

    // Top-1 accuracy remains N/A in both reusable summary and completed table
    expect(screen.getAllByText('N/A').length).toBeGreaterThan(0);
    expect(successCb).toHaveBeenCalledWith(
      expect.objectContaining({
        warnings: expect.arrayContaining([
          expect.stringContaining('Verified precomputed accuracy is unavailable for DenseNet121 on TFLITE'),
        ]),
      })
    );
  });

  it('preserves warnings through polling even if fetchBenchmarkRun has a transient initial error', async () => {
    const mockInitialRun = { id: 91, status: 'running' };
    const mockCompletedProgress = {
      run_id: 91,
      status: 'completed',
      images_completed: 100,
      total_images: 100,
      images_remaining: 0,
      percentage: 100,
      stage: 'performance_benchmarking',
      warnings: [
        'Notice: Precomputed accuracy is unavailable for DenseNet121 on TFLITE.',
      ],
    };
    const mockFinalRun = {
      id: 91,
      status: 'completed',
      warnings: [
        'Notice: Precomputed accuracy is unavailable for DenseNet121 on TFLITE.',
      ],
      measurements: [
        {
          id: 911,
          run_id: 91,
          model_id: 'densenet121',
          architecture: 'DenseNet121',
          runtime: 'tflite',
          device: 'cpu',
          status: 'completed',
          latency_mean_ms: 98.4,
          top1_accuracy: null,
        },
      ],
    };

    vi.spyOn(client, 'runBenchmarkComparison').mockResolvedValue(mockInitialRun);
    vi.spyOn(client, 'fetchBenchmarkProgress').mockResolvedValue(mockCompletedProgress);
    // First attempt fails transiently, second succeeds
    vi.spyOn(client, 'fetchBenchmarkRun')
      .mockRejectedValueOnce(new Error('Transient DB read lock'))
      .mockResolvedValue(mockFinalRun);

    render(
      <BenchmarkRunner
        selectedConfigs={{ densenet121: ['tflite'] }}
        onBenchmarkSuccess={() => {}}
      />
    );

    const runBtn = screen.getByRole('button', { name: /Run Benchmark/i });
    fireEvent.click(runBtn);

    // Polling retries and resolves after transient error
    await waitFor(() => {
      expect(screen.getByText(/Benchmark completed successfully! Run ID #91/i)).toBeInTheDocument();
    });

    const resultWarning = document.getElementById('benchmark-result-warning');
    expect(resultWarning).toBeInTheDocument();
    expect(resultWarning).toHaveTextContent(/Precomputed accuracy is unavailable for DenseNet121 on TFLITE/i);
    expect(screen.getByText('98.4 ms')).toBeInTheDocument();
  });
});


