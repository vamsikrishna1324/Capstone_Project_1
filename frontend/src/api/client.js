/**
 * Reusable REST API Client for Automated Image Model Benchmarking.
 *
 * Configures base URL from environment variable VITE_API_BASE_URL with safe
 * local fallback. Provides normalized error handling and typed response parsing.
 */

const RAW_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
export const API_BASE = RAW_BASE_URL.replace(/\/+$/, '');

/**
 * Perform an HTTP request with error parsing.
 * @param {string} endpoint - API path (e.g. '/api/health')
 * @param {RequestInit} [options] - Fetch options
 * @returns {Promise<any>}
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const headers = {
    Accept: 'application/json',
    ...(options.headers || {}),
  };

  if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorMessage = `API Error (${response.status} ${response.statusText})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          if (Array.isArray(errorJson.detail)) {
            // Pydantic validation error array
            errorMessage = errorJson.detail
              .map((err) => `${err.loc ? err.loc.join('.') : 'Field'}: ${err.msg}`)
              .join('; ');
          } else if (typeof errorJson.detail === 'string') {
            errorMessage = errorJson.detail;
          }
        }
      } catch {
        // Response wasn't JSON
      }
      const error = new Error(errorMessage);
      error.status = response.status;
      throw error;
    }

    return await response.json();
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(
        `Cannot connect to backend API server at ${url}. Please verify FastAPI is running on port 8000.`
      );
    }
    throw err;
  }
}

/**
 * Health check endpoint.
 */
export async function fetchHealth() {
  return request('/api/health');
}

/**
 * Supported models catalog and artifact statuses.
 */
export async function fetchModels() {
  return request('/api/models');
}

/**
 * Query saved benchmark results with filtering and pagination.
 */
export async function fetchResults({
  modelIds = [],
  runtimes = [],
  device = '',
  runId = null,
  limit = 50,
  offset = 0,
} = {}) {
  const params = new URLSearchParams();
  if (modelIds.length) params.set('model_ids', modelIds.join(','));
  if (runtimes.length) params.set('runtimes', runtimes.join(','));
  if (device) params.set('device', device);
  if (runId !== null && runId !== undefined) params.set('run_id', String(runId));
  params.set('limit', String(limit));
  params.set('offset', String(offset));

  const query = params.toString();
  return request(`/api/results${query ? `?${query}` : ''}`);
}

/**
 * Run a multi-model benchmark comparison and persist results.
 */
export async function runBenchmarkComparison(payload) {
  return request('/api/benchmarks', {
    method: 'POST',
    body: payload,
  });
}

/**
 * Retrieve a specific comparison run and its measurements by ID.
 */
export async function fetchBenchmarkRun(runId) {
  return request(`/api/benchmarks/${runId}`);
}

/**
 * Retrieve real-time intermediate evaluation progress by run ID.
 */
export async function fetchBenchmarkProgress(runId) {
  return request(`/api/benchmarks/${runId}/progress`);
}

/**
 * Request multi-criteria model deployment recommendation.
 */
export async function fetchRecommendation(payload) {
  return request('/api/recommendations', {
    method: 'POST',
    body: payload,
  });
}

/**
 * Retrieve verified Edge Impulse profiling hardware targets.
 */
export async function fetchEdgeImpulseTargets() {
  return request('/api/edge-impulse/targets');
}

/**
 * Profile or reuse verified Edge Impulse estimation for a supported TFLite model.
 */
export async function profileModelEdgeImpulse(payload) {
  return request('/api/edge-impulse/profile', {
    method: 'POST',
    body: payload,
  });
}
