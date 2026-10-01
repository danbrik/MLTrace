import { afterEach, expect, it, vi } from 'vitest';
import { loadInferenceExport } from './load';
import { setActiveProject } from '../api';
import type { TestingRunPlotSeriesPage } from '../types';

const options = { samplingRate: 'original', method: 'mean', utc: false } as const;
const run = { id: 1, name: 'A', status: 'finished' };
const point = (position: number) => ({ position, timestamp: `2025-09-16T00:0${position}:00`, value: position + 0.5 });
const page = (extra: Partial<TestingRunPlotSeriesPage> = {}): TestingRunPlotSeriesPage => ({
  testing_run_id: 1, score_series: 'score', result_revision: 9, total: 2,
  points: [point(0)], next_timestamp: point(0).timestamp, next_position: 0, ...extra,
});
const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body });
afterEach(() => { setActiveProject(null); vi.unstubAllGlobals(); });

it('loads every page with the revision/cursor and project pinned, in selection order', async () => {
  const fetch = vi.fn().mockImplementationOnce(async () => {
    setActiveProject('another-project');
    return ok(page());
  }).mockResolvedValueOnce(ok(page({ points: [point(1)], next_timestamp: null, next_position: null })))
    .mockResolvedValueOnce(ok(page({ testing_run_id: 2, total: 1, points: [point(0)], next_timestamp: null, next_position: null })));
  vi.stubGlobal('fetch', fetch);
  const progress = vi.fn();
  const result = await loadInferenceExport([run, { ...run, id: 2, name: 'B' }], options, 'export-project', undefined, progress);
  expect(result.columns.map((column) => column.values)).toEqual([['2025.09.16 00:00:00', '2025.09.16 00:01:00'], [0.5, 1.5], [0.5, null]]);
  expect(fetch).toHaveBeenCalledTimes(3);
  const cursor = new URL(fetch.mock.calls[1][0], 'http://localhost').searchParams;
  expect(cursor.get('after_position')).toBe('0');
  expect(cursor.get('after_timestamp')).toBe(point(0).timestamp);
  expect(cursor.get('expected_result_revision')).toBe('9');
  expect(cursor.get('score_series')).toBe('score');
  expect(fetch.mock.calls.every(([, init]) => init.headers['X-MLTrace-Project-ID'] === 'export-project')).toBe(true);
  expect(progress.mock.calls).toEqual([[1], [2]]);
});

it('surfaces a revision conflict without returning a partial export', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(ok(page())).mockResolvedValueOnce({ ok: false, status: 409, json: async () => ({ detail: 'Inference results changed. Start the export again.' }) });
  vi.stubGlobal('fetch', fetch);
  await expect(loadInferenceExport([run], options, 'project')).rejects.toThrow('Inference "A" (#1): Inference results changed');
});

it('rejects incomplete, empty, unfinished and failed sources', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(ok(page({ next_timestamp: null, next_position: null })))
    .mockResolvedValueOnce(ok(page({ total: 0, points: [], next_timestamp: null, next_position: null })))
    .mockRejectedValueOnce(new Error('Network unavailable'));
  vi.stubGlobal('fetch', fetch);
  await expect(loadInferenceExport([run], options, 'project')).rejects.toThrow('could not be loaded completely');
  await expect(loadInferenceExport([run], options, 'project')).rejects.toThrow('No stored inference points');
  await expect(loadInferenceExport([{ ...run, status: 'running' }], options, 'project')).rejects.toThrow('no longer finished');
  await expect(loadInferenceExport([run], options, 'project')).rejects.toThrow('Network unavailable');
});

it('stops an obsolete preview before making another source request', async () => {
  const controller = new AbortController();
  const fetch = vi.fn().mockResolvedValue(ok(page({ total: 1, next_timestamp: null, next_position: null })));
  vi.stubGlobal('fetch', fetch);
  await expect(loadInferenceExport([run, { ...run, id: 2 }], options, 'project', controller.signal, () => controller.abort())).rejects.toThrow();
  expect(fetch).toHaveBeenCalledTimes(1);
});
