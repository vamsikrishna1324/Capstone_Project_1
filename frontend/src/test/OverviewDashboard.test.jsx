import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import OverviewDashboard from '../components/OverviewDashboard';

describe('OverviewDashboard Component', () => {
  const mockModels = [
    { model_id: 'mobilenet_v3_small', display_name: 'MobileNetV3-Small' },
    { model_id: 'resnet50', display_name: 'ResNet50' },
  ];

  const mockResults = [
    {
      id: 1,
      run_id: 10,
      model_id: 'mobilenet_v3_small',
      architecture: 'MobileNetV3-Small',
      runtime: 'onnx',
      device: 'cpu',
      provenance: 'live_execution',
      created_at: '2026-10-18T10:00:00Z',
      latency: { mean_ms: 1.48 },
      accuracy: { top1_accuracy: 0.6767 },
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
      created_at: '2026-10-18T12:00:00Z',
      latency: { mean_ms: 21477.0 },
      accuracy: null,
      model_size: { file_size_mb: 4.9 },
    },
  ];

  it('renders all 5 KPI summary cards with approved titles and does NOT render the comparison matrix', () => {
    render(
      <OverviewDashboard
        models={mockModels}
        results={mockResults}
        totalResults={2}
        onNavigate={vi.fn()}
      />
    );

    // Approved 5 KPI cards are rendered
    expect(screen.getByText('Supported Vision Models')).toBeInTheDocument();
    expect(screen.getByText('Saved Runs')).toBeInTheDocument();
    expect(screen.getByText('Lowest Measured Latency')).toBeInTheDocument();
    expect(screen.getByText('Highest Validated Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Smallest Model Size')).toBeInTheDocument();

    // Verifies model-matrix selection checkboxes are strictly absent from Overview
    expect(screen.queryByText(/Select models and runtimes to compare/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Compare Selected/i })).not.toBeInTheDocument();
  });

  it('renders the How to Use EdgeBench section with all 5 numbered steps', () => {
    render(
      <OverviewDashboard
        models={mockModels}
        results={mockResults}
        totalResults={2}
        onNavigate={vi.fn()}
      />
    );

    expect(screen.getByText('How to Use EdgeBench')).toBeInTheDocument();
    expect(
      screen.getByText('Follow these simple steps to benchmark and select the best image classification model for your edge deployment.')
    ).toBeInTheDocument();

    // 5 numbered steps
    expect(screen.getByText('Compare Models')).toBeInTheDocument();
    expect(screen.getByText(/Go to Model Comparison to view supported models/i)).toBeInTheDocument();

    expect(screen.getByText('Run Benchmark')).toBeInTheDocument();
    expect(screen.getByText(/Navigate to Run Benchmark, choose a model/i)).toBeInTheDocument();

    expect(screen.getByText('View Results')).toBeInTheDocument();
    expect(screen.getByText(/Monitor progress and view available metrics/i)).toBeInTheDocument();

    expect(screen.getByText('Get Recommendations')).toBeInTheDocument();
    expect(screen.getByText(/Use Recommendations to identify suitable models/i)).toBeInTheDocument();

    expect(screen.getByText('Check Run History')).toBeInTheDocument();
    expect(screen.getByText(/Visit Run History to review previous benchmark sessions/i)).toBeInTheDocument();
  });

  it('renders all four quick action cards and navigates to the corresponding tab on click', () => {
    const handleNavigate = vi.fn();
    render(
      <OverviewDashboard
        models={mockModels}
        results={mockResults}
        totalResults={2}
        onNavigate={handleNavigate}
      />
    );

    expect(screen.getByText('Platform Workspaces & Navigation')).toBeInTheDocument();

    const compareBtn = screen.getByRole('button', { name: /Open Comparison Matrix/i });
    fireEvent.click(compareBtn);
    expect(handleNavigate).toHaveBeenCalledWith('compare');

    const benchmarkBtn = screen.getByRole('button', { name: /Launch Benchmark Runner/i });
    fireEvent.click(benchmarkBtn);
    expect(handleNavigate).toHaveBeenCalledWith('benchmark');

    const recsBtn = screen.getByRole('button', { name: /View Recommendations/i });
    fireEvent.click(recsBtn);
    expect(handleNavigate).toHaveBeenCalledWith('recommendations');

    const historyBtn = screen.getByRole('button', { name: /Explore Run History/i });
    fireEvent.click(historyBtn);
    expect(handleNavigate).toHaveBeenCalledWith('history');
  });

  it('strictly excludes Recent Benchmark Sessions and System Connectivity from Overview main content', () => {
    render(
      <OverviewDashboard
        models={mockModels}
        results={mockResults}
        totalResults={2}
        onNavigate={vi.fn()}
      />
    );

    expect(screen.queryByText('Recent Benchmark Sessions')).not.toBeInTheDocument();
    expect(screen.queryByText('System Connectivity')).not.toBeInTheDocument();
    expect(screen.queryByText('No Benchmark Sessions Recorded Yet')).not.toBeInTheDocument();
  });
});
