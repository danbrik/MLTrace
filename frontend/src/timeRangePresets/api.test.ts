import { afterEach, expect, it, vi } from 'vitest';
import { createTimeRangePreset, deleteTimeRangePreset, listTimeRangePresets, updateTimeRangePreset } from '../api';

afterEach(() => vi.unstubAllGlobals());
it('uses project-scoped API URLs and explicit project headers for every operation', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => [] });
  vi.stubGlobal('fetch', fetch);
  const input = { name: 'Normal', start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' };
  await listTimeRangePresets('project-a');
  await createTimeRangePreset('project-a', input);
  await updateTimeRangePreset('project-a', 7, input);
  fetch.mockResolvedValueOnce({ ok: true, status: 204 });
  await deleteTimeRangePreset('project-b', 7);
  expect(fetch.mock.calls.map(([url]) => url)).toEqual(['/api/time-range-presets', '/api/time-range-presets', '/api/time-range-presets/7', '/api/time-range-presets/7']);
  expect(fetch.mock.calls.map(([, options]) => options.headers['X-MLTrace-Project-ID'])).toEqual(['project-a', 'project-a', 'project-a', 'project-b']);
  expect(fetch.mock.calls.map(([, options]) => options.method)).toEqual([undefined, 'POST', 'PUT', 'DELETE']);
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual(input);
});
