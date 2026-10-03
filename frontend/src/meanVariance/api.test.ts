import { afterEach, expect, it, vi } from 'vitest';
import { initialConfig } from './helpers';
import { previewMeanVariance, createMeanVarianceRun, listMeanVarianceRuns, getMeanVarianceRun, getMeanVarianceResults, abortMeanVarianceRun, deleteMeanVarianceRun, getMeanVarianceLog, meanVarianceArtifactUrl } from '../api';
afterEach(() => vi.unstubAllGlobals());
it('scopes all comparison requests and PNG URLs to the selected project', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  vi.stubGlobal('fetch', fetch);
  await previewMeanVariance(initialConfig, 'project A'); await createMeanVarianceRun(initialConfig, 'project A');
  await listMeanVarianceRuns('project A'); await getMeanVarianceRun(7, 'project A');
  await getMeanVarianceResults(7, 'project A'); await getMeanVarianceLog(7, 'project A');
  await abortMeanVarianceRun(7, 'project A'); await deleteMeanVarianceRun(7, 'project A');
  expect(fetch.mock.calls.map(([url]) => url)).toEqual(['preview', 'runs', 'runs', 'runs/7', 'runs/7/results', 'runs/7/log', 'runs/7/abort', 'runs/7'].map(path => `/api/mean-variance-analysis/${path}`));
  expect(fetch.mock.calls.every(([, options]) => options.headers['X-MLTrace-Project-ID'] === 'project A')).toBe(true);
  expect(fetch.mock.calls.at(-1)?.[1].method).toBe('DELETE');
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toMatchObject({ version: 2, sampling_rate: 1, pairs: initialConfig.pairs });
  expect(meanVarianceArtifactUrl(7, 'variance_comparison.png', 'project A', true)).toBe('/api/mean-variance-analysis/runs/7/artifacts/variance_comparison.png?download=true&project_id=project%20A');
  expect(meanVarianceArtifactUrl(7, 'variance_difference.png', 'project A', true)).toBe('/api/mean-variance-analysis/runs/7/artifacts/variance_difference.png?download=true&project_id=project%20A');
});
