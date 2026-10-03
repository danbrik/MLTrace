import { afterEach, expect, it, vi } from 'vitest';
import { createResolutionSensitivityRun, getResolutionSensitivityRun, listResolutionSensitivityRuns, setActiveProject } from '../api';

afterEach(() => { vi.unstubAllGlobals(); setActiveProject(null); });
it('keeps analysis requests bound to the supplied project after navigation', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => [] });
  vi.stubGlobal('fetch', fetch);
  setActiveProject('other-project');
  const payload = { training_dataset_id: 1, pipeline_ids: [1, 2, 3, 4], intervals: [], samples_per_interval: 15 };
  await createResolutionSensitivityRun(payload, 'analysis-project');
  await listResolutionSensitivityRuns('analysis-project');
  await getResolutionSensitivityRun(7, 'analysis-project');
  expect(fetch.mock.calls.map(([, options]) => options.headers['X-MLTrace-Project-ID'])).toEqual([
    'analysis-project', 'analysis-project', 'analysis-project',
  ]);
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual(payload);
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    '/api/resolution-sensitivity-runs', '/api/resolution-sensitivity-runs', '/api/resolution-sensitivity-runs/7',
  ]);
});
