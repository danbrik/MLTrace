import { afterEach, expect, it, vi } from 'vitest';
import { defaultConfig, defaultPlot } from './helpers';
import { previewTemporalDifference, createTemporalDifferenceRun, listTemporalDifferenceRuns, getTemporalDifferenceRun, getTemporalDifferenceSummary,
  getTemporalDifferencePairs, saveTemporalDifferencePlot, getTemporalDifferenceLog, abortTemporalDifferenceRun, deleteTemporalDifferenceRun, temporalDifferenceCsvUrl } from '../api';
afterEach(() => vi.unstubAllGlobals());
it('scopes all saved-result operations and exports to the selected project', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  vi.stubGlobal('fetch', fetch);
  await previewTemporalDifference(defaultConfig, 'project A'); await createTemporalDifferenceRun(defaultConfig, 'project A');
  await listTemporalDifferenceRuns('project A'); await getTemporalDifferenceRun(3, 'project A');
  await getTemporalDifferenceSummary(3, 'project A'); await getTemporalDifferencePairs(3, 50, 'comparison', '2', 'project A');
  await saveTemporalDifferencePlot(3, defaultPlot, 'project A'); await getTemporalDifferenceLog(3, 'project A');
  await abortTemporalDifferenceRun(3, 'project A'); await deleteTemporalDifferenceRun(3, 'project A');
  expect(fetch.mock.calls.every(([, options]) => options.headers['X-MLTrace-Project-ID'] === 'project A')).toBe(true);
  expect(fetch.mock.calls[5][0]).toBe('/api/temporal-difference/runs/3/pairs?offset=50&limit=50&unit=percent&role=comparison&delta=2');
  expect(fetch.mock.calls[6][1].method).toBe('PUT');
  expect(JSON.parse(fetch.mock.calls[6][1].body)).toEqual(defaultPlot);
  expect(temporalDifferenceCsvUrl(3, 'pairs', 'project A')).toBe('/api/temporal-difference/runs/3/csv/pairs?unit=percent&project_id=project%20A');
});
