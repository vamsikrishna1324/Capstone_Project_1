import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import Recommendations from '../components/Recommendations';
import * as client from '../api/client';

describe('Recommendations Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('does not call recommendation API or render recommendation cards on initial page load', () => {
    const fetchSpy = vi.spyOn(client, 'fetchRecommendation');

    render(<Recommendations selectedModelIds={[]} />);

    // API must NOT be called on mount
    expect(fetchSpy).not.toHaveBeenCalled();

    // Controls must be rendered
    expect(screen.getByText('1. Select Operational Priority Preset')).toBeInTheDocument();
    expect(screen.getByText('2. Deployment Hardware Constraints (Optional Thresholds)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate recommendation/i })).toBeInTheDocument();

    // Unprompted configuration message must be visible
    expect(screen.getByText('Configure Requirements and Generate Recommendations')).toBeInTheDocument();

    // Recommendation results, rankings table, stale banner must NOT be present initially
    expect(screen.queryByText(/Top-ranked match/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/#1 • Best Match/i)).not.toBeInTheDocument();
    expect(screen.queryByText('All Evaluated Candidate Rankings')).not.toBeInTheDocument();
    expect(screen.queryByText('Visual Comparison of Recommended Candidates')).not.toBeInTheDocument();
    expect(screen.queryByText(/Recommendations Stale/i)).not.toBeInTheDocument();
  });

  it('displays recommendations only after clicking Generate Recommendation', async () => {
    const populatedResponse = {
      recommended_model: 'mobilenet_v3_small',
      recommended_display_name: 'MobileNetV3-Small',
      recommended_runtime: 'onnx',
      composite_score: 87.5,
      explanation: 'MobileNetV3-Small [ONNX] delivers the best balanced performance across latency and footprint.',
      trade_off_analysis: 'Sub-2ms execution with modest ImageNet accuracy trade-off.',
      criteria_weights_applied: { accuracy: 0.35, latency: 0.30, memory: 0.20, model_size: 0.15 },
      constraints_applied: {},
      top_recommendations: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          device: 'cpu',
          composite_score: 87.5,
          raw_metrics: {
            accuracy: 0.6767,
            latency: 1.48,
            memory: 652.6,
            model_size: 9.72,
          },
          subscores: { accuracy: 83.7, latency: 100.0, memory: 100.0, model_size: 100.0 },
          missing_metrics: [],
          data_complete: true,
          eligible: true,
          disqualification_reasons: [],
          satisfied_constraints: [],
          rationale: 'Top-ranked match scoring 87.5/100 aligned with your priorities.',
          trade_offs: 'Outranks runner-up ResNet50.',
        },
      ],
      rankings: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          device: 'cpu',
          composite_score: 87.5,
          raw_metrics: {
            accuracy: 0.6767,
            latency: 1.48,
            memory: 652.6,
            model_size: 9.72,
          },
          eligible: true,
          disqualification_reasons: [],
        },
      ],
    };

    const fetchSpy = vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(populatedResponse);

    render(<Recommendations selectedModelIds={['mobilenet_v3_small']} />);

    expect(fetchSpy).not.toHaveBeenCalled();

    // Click Generate Recommendation
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    expect(fetchSpy).toHaveBeenCalledTimes(1);

    await waitFor(() => {
      expect(screen.getByText(/#1 • Best Match/i)).toBeInTheDocument();
    });

    expect(screen.getByText('MobileNetV3-Small [ONNX]')).toBeInTheDocument();
    expect(screen.getAllByText('87.5').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('All Evaluated Candidate Rankings')).toBeInTheDocument();
    expect(screen.getByText('Visual Comparison of Recommended Candidates')).toBeInTheDocument();
  });

  it('sends selected priority preset and hardware constraints to backend', async () => {
    const fetchSpy = vi.spyOn(client, 'fetchRecommendation').mockResolvedValue({
      recommended_model: 'mobilenet_v3_small',
      top_recommendations: [],
      rankings: [],
      criteria_weights_applied: {},
      constraints_applied: {},
      explanation: 'Done',
      trade_off_analysis: 'Done',
    });

    render(<Recommendations selectedModelIds={['mobilenet_v3_small', 'resnet50']} />);

    // Select Speed Optimized preset
    fireEvent.click(screen.getByRole('button', { name: /speed optimized/i }));

    // Fill constraints
    fireEvent.change(document.getElementById('input-constraint-latency'), { target: { value: '25' } });
    fireEvent.change(document.getElementById('input-constraint-memory'), { target: { value: '500' } });
    fireEvent.change(document.getElementById('input-constraint-accuracy'), { target: { value: '75' } });

    // Click generate
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    expect(fetchSpy).toHaveBeenCalledWith({
      model_ids: ['mobilenet_v3_small', 'resnet50'],
      priority_preset: 'speed_optimized',
      weights: undefined,
      constraints: {
        max_latency_ms: 25,
        max_memory_mb: 500,
        max_model_size_mb: undefined,
        min_accuracy: 0.75,
      },
    });
  });

  it('rejects custom weights when all weights are zero with inline validation message', async () => {
    const fetchSpy = vi.spyOn(client, 'fetchRecommendation');

    render(<Recommendations selectedModelIds={[]} />);

    // Select Custom Weights preset
    fireEvent.click(screen.getByRole('button', { name: /custom weights/i }));

    // Set all 4 criteria sliders to 0
    fireEvent.change(document.getElementById('slider-weight-accuracy'), { target: { value: '0' } });
    fireEvent.change(document.getElementById('slider-weight-latency'), { target: { value: '0' } });
    fireEvent.change(document.getElementById('slider-weight-memory'), { target: { value: '0' } });
    fireEvent.change(document.getElementById('slider-weight-model_size'), { target: { value: '0' } });

    // Click generate
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    // Must NOT call backend API
    expect(fetchSpy).not.toHaveBeenCalled();

    // Must show inline validation error message
    expect(screen.getByText('Set at least one criterion weight above zero.')).toBeInTheDocument();
  });

  it('allows individual zero custom weights and generates recommendations', async () => {
    const fetchSpy = vi.spyOn(client, 'fetchRecommendation').mockResolvedValue({
      recommended_model: 'resnet50',
      top_recommendations: [],
      rankings: [],
      criteria_weights_applied: { accuracy: 0.8, latency: 0.0, memory: 0.2, model_size: 0.0 },
      constraints_applied: {},
      explanation: 'Evaluated with custom weights',
      trade_off_analysis: 'Latency and size ignored',
    });

    render(<Recommendations selectedModelIds={[]} />);

    fireEvent.click(screen.getByRole('button', { name: /custom weights/i }));

    // Set latency and model_size to 0, but accuracy and memory positive
    fireEvent.change(document.getElementById('slider-weight-latency'), { target: { value: '0' } });
    fireEvent.change(document.getElementById('slider-weight-model_size'), { target: { value: '0' } });
    fireEvent.change(document.getElementById('slider-weight-accuracy'), { target: { value: '0.8' } });
    fireEvent.change(document.getElementById('slider-weight-memory'), { target: { value: '0.2' } });

    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        weights: {
          accuracy: 0.8,
          latency: 0,
          memory: 0.2,
          model_size: 0,
        },
      })
    );
  });

  it('invalidates results and marks them stale when preset, weights, or constraints change after generating', async () => {
    const mockData = {
      recommended_model: 'mobilenet_v3_small',
      top_recommendations: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          composite_score: 87.5,
          raw_metrics: { accuracy: 0.67, latency: 1.48, memory: 650, model_size: 9.7 },
          eligible: true,
          satisfied_constraints: [],
        },
      ],
      rankings: [],
      criteria_weights_applied: {},
      constraints_applied: {},
      explanation: 'Top match',
      trade_off_analysis: 'Balanced',
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(mockData);

    render(<Recommendations selectedModelIds={[]} />);

    // Initial generate
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText('MobileNetV3-Small [ONNX]')).toBeInTheDocument();
    });

    // Now change preset to Accuracy First
    fireEvent.click(screen.getByRole('button', { name: /accuracy first/i }));

    // Stale banner must appear
    expect(screen.getByText(/Configuration Modified — Recommendations Stale/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Requirements have changed since recommendations were generated/i)
    ).toBeInTheDocument();

    // Previous recommendation cards must be hidden while stale
    expect(screen.queryByText('MobileNetV3-Small [ONNX]')).not.toBeInTheDocument();
    expect(screen.queryByText('Visual Comparison of Recommended Candidates')).not.toBeInTheDocument();

    // Click Regenerate Recommendations button
    fireEvent.click(screen.getByRole('button', { name: /regenerate recommendations/i }));

    await waitFor(() => {
      expect(screen.getByText('MobileNetV3-Small [ONNX]')).toBeInTheDocument();
    });

    expect(screen.queryByText(/Configuration Modified — Recommendations Stale/i)).not.toBeInTheDocument();
  });

  it('renders up to three recommendation cards with rank, score, metrics, rationale, and satisfied constraints', async () => {
    const threeCandidatesResponse = {
      recommended_model: 'mobilenet_v3_small',
      top_recommendations: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          composite_score: 88.2,
          raw_metrics: { accuracy: 0.6767, latency: 1.48, memory: 652.6, model_size: 9.72 },
          eligible: true,
          satisfied_constraints: ['Latency 1.48ms ≤ 20.00ms', 'Peak RAM 652.6MB ≤ 800.0MB'],
          rationale: 'Top-ranked match scoring 88.2/100 with sub-2ms latency.',
          trade_offs: 'Outranks runner-up ResNet50.',
        },
        {
          rank: 2,
          model_id: 'resnet50',
          display_name: 'ResNet50',
          runtime: 'onnx',
          composite_score: 75.4,
          raw_metrics: { accuracy: 0.8086, latency: 11.2, memory: 720.0, model_size: 95.0 },
          eligible: true,
          satisfied_constraints: ['Latency 11.20ms ≤ 20.00ms', 'Peak RAM 720.0MB ≤ 800.0MB'],
          rationale: 'Rank #2 eligible option delivering 80.9% Top-1 accuracy.',
          trade_offs: '+9.72ms slower latency than #1, +13.2% higher accuracy.',
        },
        {
          rank: 3,
          model_id: 'densenet121',
          display_name: 'DenseNet121',
          runtime: 'tflite',
          composite_score: 64.1,
          raw_metrics: { accuracy: 0.7498, latency: 18.5, memory: 780.0, model_size: 32.5 },
          eligible: true,
          satisfied_constraints: ['Latency 18.50ms ≤ 20.00ms', 'Peak RAM 780.0MB ≤ 800.0MB'],
          rationale: 'Rank #3 eligible option scoring 64.1/100.',
          trade_offs: 'Higher compute overhead on Host CPU.',
        },
      ],
      rankings: [],
      criteria_weights_applied: {},
      constraints_applied: {},
      explanation: 'Summary',
      trade_off_analysis: 'Comparative notes',
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(threeCandidatesResponse);

    render(<Recommendations selectedModelIds={[]} />);
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/#1 • Best Match/i)).toBeInTheDocument();
      expect(screen.getByText(/#2 • Alternative/i)).toBeInTheDocument();
      expect(screen.getByText(/#3 • Alternative/i)).toBeInTheDocument();
    });

    expect(screen.getByText('MobileNetV3-Small [ONNX]')).toBeInTheDocument();
    expect(screen.getByText('ResNet50 [ONNX]')).toBeInTheDocument();
    expect(screen.getByText('DenseNet121 [TFLITE]')).toBeInTheDocument();

    // Verify satisfied constraints are displayed
    expect(screen.getAllByText('Latency 1.48ms ≤ 20.00ms').length).toBeGreaterThanOrEqual(1);

    // Verify rationales
    expect(screen.getByText(/Top-ranked match scoring 88.2\/100 with sub-2ms latency/i)).toBeInTheDocument();
    expect(screen.getByText(/Rank #2 eligible option delivering 80.9% Top-1 accuracy/i)).toBeInTheDocument();
  });

  it('displays notice explaining when fewer than 3 candidates qualify', async () => {
    const twoCandidatesResponse = {
      recommended_model: 'mobilenet_v3_small',
      top_recommendations: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          composite_score: 88.2,
          raw_metrics: { accuracy: 0.6767, latency: 1.48, memory: 652.6, model_size: 9.72 },
          eligible: true,
          satisfied_constraints: ['Latency 1.48ms ≤ 15.00ms'],
          rationale: 'Top match.',
          trade_offs: 'Sole fast option.',
        },
        {
          rank: 2,
          model_id: 'resnet50',
          display_name: 'ResNet50',
          runtime: 'onnx',
          composite_score: 75.4,
          raw_metrics: { accuracy: 0.8086, latency: 11.2, memory: 720.0, model_size: 95.0 },
          eligible: true,
          satisfied_constraints: ['Latency 11.20ms ≤ 15.00ms'],
          rationale: 'Runner up.',
          trade_offs: 'Higher accuracy.',
        },
      ],
      rankings: [
        { rank: 1, eligible: true, model_id: 'mobilenet_v3_small', display_name: 'MobileNetV3-Small', runtime: 'onnx' },
        { rank: 2, eligible: true, model_id: 'resnet50', display_name: 'ResNet50', runtime: 'onnx' },
        { rank: 3, eligible: false, model_id: 'densenet121', display_name: 'DenseNet121', runtime: 'tflite', disqualification_reasons: ['Latency exceeds 15ms'] },
      ],
      criteria_weights_applied: {},
      constraints_applied: {},
      explanation: 'Filtered by latency',
      trade_off_analysis: 'Trade-offs',
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(twoCandidatesResponse);

    render(<Recommendations selectedModelIds={[]} />);
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/Showing all/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Showing all/i)).toHaveTextContent('Showing all 2 eligible candidates satisfying your specified constraints');
  });

  it('renders No Eligible Model Found banner when zero candidates qualify', async () => {
    const disqualifiedResponse = {
      recommended_model: null,
      top_recommendations: [],
      explanation: 'All candidate models exceeded the max latency threshold of 1.0 ms.',
      trade_off_analysis: 'Consider relaxing constraints to allow qualified models.',
      criteria_weights_applied: { accuracy: 0.35, latency: 0.30, memory: 0.20, model_size: 0.15 },
      constraints_applied: { max_latency_ms: 1.0 },
      rankings: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          composite_score: 0.0,
          raw_metrics: { accuracy: 0.6767, latency: 1.48, memory: 652.6, model_size: 9.72 },
          eligible: false,
          disqualification_reasons: ['Latency 1.48 ms exceeds max limit 1.0 ms'],
        },
      ],
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(disqualifiedResponse);

    render(<Recommendations selectedModelIds={['mobilenet_v3_small']} />);
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText('No Eligible Model Found')).toBeInTheDocument();
    });

    expect(screen.getByText(/exceeded the max latency threshold/i)).toBeInTheDocument();
  });

  it('renders neutral informational empty state when database has no benchmark measurements', async () => {
    const emptyResponse = {
      recommended_model: null,
      top_recommendations: [],
      explanation: 'No benchmark measurements are currently available in the database.',
      trade_off_analysis: 'No candidate models evaluated.',
      criteria_weights_applied: { accuracy: 0.35, latency: 0.30, memory: 0.20, model_size: 0.15 },
      constraints_applied: {},
      rankings: [],
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(emptyResponse);

    render(<Recommendations selectedModelIds={[]} />);
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText('No Benchmark Data Available')).toBeInTheDocument();
    });

    expect(screen.queryByText('All Evaluated Candidate Rankings')).not.toBeInTheDocument();
    expect(screen.queryByText('No Eligible Model Found')).not.toBeInTheDocument();
  });

  it('renders visual comparison chart with direction badges and handles unmeasured data with N/A', async () => {
    const responseWithMissing = {
      recommended_model: 'mobilenet_v3_small',
      top_recommendations: [
        {
          rank: 1,
          model_id: 'mobilenet_v3_small',
          display_name: 'MobileNetV3-Small',
          runtime: 'onnx',
          composite_score: 87.5,
          raw_metrics: {
            accuracy: 0.6767,
            latency: 1.48,
            memory: 652.6,
            model_size: 9.72,
          },
          eligible: true,
          satisfied_constraints: [],
        },
        {
          rank: 2,
          model_id: 'resnet50',
          display_name: 'ResNet50',
          runtime: 'pytorch',
          composite_score: 62.1,
          raw_metrics: {
            accuracy: 0.8086,
            latency: 13.02,
            memory: 829.1,
            model_size: null, // PyTorch null model size
          },
          eligible: true,
          satisfied_constraints: [],
        },
      ],
      rankings: [],
      criteria_weights_applied: {},
      constraints_applied: {},
      explanation: 'Done',
      trade_off_analysis: 'Done',
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(responseWithMissing);

    render(<Recommendations selectedModelIds={[]} />);
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText('Visual Comparison of Recommended Candidates')).toBeInTheDocument();
    });

    // Verify direction labels
    expect(screen.getAllByText('Lower is better ↓').length).toBe(3); // latency, memory, size
    expect(screen.getByText('Higher is better ↑')).toBeInTheDocument(); // accuracy

    // Verify N/A badge for missing PyTorch model size
    expect(screen.getAllByText('N/A (Not measured)').length).toBeGreaterThanOrEqual(1);
  });

  it('removes the bottom Edge Impulse analytical section entirely', async () => {
    render(
      <Recommendations
        selectedModelIds={[]}
        savedResults={[
          {
            id: 1,
            run_id: 1,
            model_id: 'mobilenet_v3_small',
            runtime: 'tflite',
            device: 'raspberry-pi-4',
            provenance: 'edge_impulse_estimation',
            latency_mean_ms: 21477.0,
          },
        ]}
      />
    );

    // Verify the Edge Impulse section does NOT exist
    expect(document.getElementById('edge-impulse-analytical-section')).toBeNull();
    expect(screen.queryByText(/Edge Impulse Analytical Estimates/i)).not.toBeInTheDocument();
  });

  it('selects Accuracy First preset and displays dominant accuracy weights and rationale', async () => {
    const fetchSpy = vi.spyOn(client, 'fetchRecommendation').mockResolvedValue({
      recommended_model: 'resnet50',
      top_recommendations: [
        {
          rank: 1,
          model_id: 'resnet50',
          display_name: 'ResNet50',
          runtime: 'onnx',
          composite_score: 81.8,
          raw_metrics: { accuracy: 0.8086, latency: 13.0, memory: 720.0, model_size: 95.0 },
          eligible: true,
          satisfied_constraints: [],
          rationale: 'Top-ranked match scoring 81.8/100 aligned with requested priorities (accuracy (75%), latency (10%), memory (8%), model size (8%)).',
          trade_offs: 'Outranks runner-up MobileNetV3.',
        },
      ],
      rankings: [],
      criteria_weights_applied: { accuracy: 0.75, latency: 0.10, memory: 0.075, model_size: 0.075 },
      constraints_applied: {},
      explanation: 'ResNet50 on ONNX is the top recommended deployment configuration scoring 81.8/100 based on accuracy (75%).',
      trade_off_analysis: 'Dominates in verified accuracy.',
    });

    render(<Recommendations selectedModelIds={[]} />);

    // Select Accuracy First
    fireEvent.click(screen.getByRole('button', { name: /accuracy first/i }));

    // Click Generate Recommendation
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        priority_preset: 'accuracy_first',
      })
    );

    await waitFor(() => {
      expect(screen.getByText('ResNet50 [ONNX]')).toBeInTheDocument();
    });

    expect(screen.getByText(/accuracy \(75%\)/i)).toBeInTheDocument();
  });

  it('renders distinct rationales for each recommendation card without repeating candidate 1 rationale', async () => {
    const multiCandidatesResponse = {
      recommended_model: 'resnet50',
      top_recommendations: [
        {
          rank: 1,
          model_id: 'resnet50',
          display_name: 'ResNet50',
          runtime: 'onnx',
          composite_score: 83.8,
          raw_metrics: { accuracy: 0.8085, latency: 11.38, memory: 932.3, model_size: 97.42 },
          eligible: true,
          satisfied_constraints: [],
          rationale: 'Top-ranked match scoring 83.8/100 aligned with accuracy (75%). Delivers 80.8% Top-1 accuracy.',
          trade_offs: 'Outranks runner-up ResNet50 TFLite.',
        },
        {
          rank: 2,
          model_id: 'resnet50',
          display_name: 'ResNet50',
          runtime: 'tflite',
          composite_score: 81.7,
          raw_metrics: { accuracy: 0.8086, latency: 90.5, memory: 788.1, model_size: 48.74 },
          eligible: true,
          satisfied_constraints: [],
          rationale: 'Rank #2 eligible option scoring 81.7/100 based on accuracy (75%). Features 80.9% Top-1 accuracy.',
          trade_offs: '+79.12ms slower latency than #1.',
        },
        {
          rank: 3,
          model_id: 'densenet121',
          display_name: 'DenseNet121',
          runtime: 'onnx',
          composite_score: 59.1,
          raw_metrics: { accuracy: 0.7444, latency: 25.01, memory: 508.6, model_size: 30.84 },
          eligible: true,
          satisfied_constraints: [],
          rationale: 'Rank #3 eligible option scoring 59.1/100 based on accuracy (75%). Features 74.4% Top-1 accuracy.',
          trade_offs: '+13.63ms slower latency than #1.',
        },
      ],
      rankings: [],
      criteria_weights_applied: { accuracy: 0.75, latency: 0.10, memory: 0.075, model_size: 0.075 },
      constraints_applied: {},
      explanation: 'ResNet50 on ONNX is the overall winner scoring 83.8/100.',
      trade_off_analysis: 'High accuracy edge trade-off.',
    };

    vi.spyOn(client, 'fetchRecommendation').mockResolvedValue(multiCandidatesResponse);

    render(<Recommendations selectedModelIds={[]} />);
    fireEvent.click(screen.getByRole('button', { name: /generate recommendation/i }));

    await waitFor(() => {
      expect(screen.getByText(/#1 • Best Match/i)).toBeInTheDocument();
      expect(screen.getByText(/#2 • Alternative/i)).toBeInTheDocument();
      expect(screen.getByText(/#3 • Alternative/i)).toBeInTheDocument();
    });

    // Verify distinct rationales exist on the screen
    expect(screen.getByText(/Top-ranked match scoring 83.8\/100 aligned with accuracy \(75%\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Rank #2 eligible option scoring 81.7\/100 based on accuracy \(75%\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Rank #3 eligible option scoring 59.1\/100 based on accuracy \(75%\)/i)).toBeInTheDocument();

    // Verify that Card 2 does NOT repeat Candidate 1's rationale
    const card2 = document.getElementById('recommendation-card-2');
    expect(card2).not.toHaveTextContent('Top-ranked match');
    expect(card2).toHaveTextContent('Rank #2 eligible option');

    // Verify that Card 3 does NOT repeat Candidate 1's rationale
    const card3 = document.getElementById('recommendation-card-3');
    expect(card3).not.toHaveTextContent('Top-ranked match');
    expect(card3).toHaveTextContent('Rank #3 eligible option');
  });
});
