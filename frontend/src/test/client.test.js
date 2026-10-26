import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fetchEdgeImpulseTargets, profileModelEdgeImpulse } from '../api/client';

describe('API Client - Edge Impulse Integration', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('fetchEdgeImpulseTargets sends GET request to /api/edge-impulse/targets', async () => {
    const mockTargetsResponse = {
      targets: [
        {
          target_id: 'raspberry-pi-4',
          display_name: 'Raspberry Pi 4',
          supported_formats: ['tflite'],
          is_supported: true,
        },
      ],
      total_targets: 1,
    };

    globalThis.fetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockTargetsResponse,
    });

    const result = await fetchEdgeImpulseTargets();

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    const [url, options] = globalThis.fetch.mock.calls[0];
    expect(url).toContain('/api/edge-impulse/targets');
    expect(options.headers['Accept']).toBe('application/json');
    expect(result).toEqual(mockTargetsResponse);
  });

  it('profileModelEdgeImpulse sends POST request to /api/edge-impulse/profile with payload', async () => {
    const mockProfileResponse = {
      id: 77,
      status: 'success',
      model_id: 'mobilenet_v3_small',
      runtime: 'tflite',
      latency_ms: 21477.0,
      provenance: 'edge_impulse_estimation',
    };

    globalThis.fetch.mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => mockProfileResponse,
    });

    const payload = {
      model_id: 'mobilenet_v3_small',
      target_device: 'raspberry-pi-4',
      runtime: 'tflite',
      batch_size: 1,
      reuse_completed: true,
    };

    const result = await profileModelEdgeImpulse(payload);

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    const [url, options] = globalThis.fetch.mock.calls[0];
    expect(url).toContain('/api/edge-impulse/profile');
    expect(options.method).toBe('POST');
    expect(JSON.parse(options.body)).toEqual(payload);
    expect(result).toEqual(mockProfileResponse);
  });

  it('profileModelEdgeImpulse propagates error response message on HTTP failure', async () => {
    globalThis.fetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Edge Impulse profiling supports TFLite models only.' }),
    });

    await expect(
      profileModelEdgeImpulse({
        model_id: 'mobilenet_v3_small',
        target_device: 'raspberry-pi-4',
        runtime: 'onnx',
      })
    ).rejects.toThrow('Edge Impulse profiling supports TFLite models only.');
  });
});
