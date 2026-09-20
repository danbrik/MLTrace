import { afterEach, describe, expect, it, vi } from 'vitest';
import { setActiveProject, timeSeriesApi } from '../api';

function captureRequest(response: unknown) {
  const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  setActiveProject('label-import-project');
  return fetchMock;
}

afterEach(() => { setActiveProject(null); vi.unstubAllGlobals(); });

describe('label import API contract', () => {
  const file = new File(['time,sensor,label\n2026-01-01T00:00:00Z,1,normal'], 'labels.csv', { type: 'text/csv' });
  it('passes the annotation column to preview and preserves the server proposal', async () => {
    const proposal = { detected_label_column: 'label', label_split: { tags: ['anomaly'], intervals: [], counts: { train: { rows: 1, intervals: 1 } } } };
    const request = captureRequest(proposal);
    expect(await timeSeriesApi.preview(file, 'time', 'ISO8601', 'label')).toEqual(proposal);
    const [url, options] = request.mock.calls[0];
    expect(url).toBe('/api/time-series/preview');
    const body = options!.body as FormData;
    expect(body.get('label_column')).toBe('label');
    expect(body.get('timestamp_column')).toBe('time');
    expect(options!.headers).toMatchObject({ 'X-MLTrace-Project-ID': 'label-import-project' });
  });
  it('omits label parsing when automatic split is disabled', async () => {
    const request = captureRequest({});
    await timeSeriesApi.preview(file, 'time');
    expect((request.mock.calls[0][1]!.body as FormData).has('label_column')).toBe(false);
  });
  it('sends dataset and split options in one atomic import request', async () => {
    const request = captureRequest({ id: 1, split_count: 1, label_column: 'label' });
    const metadata = { name: 'Sensor', selected_columns: ['time', 'sensor'], timestamp_column: 'time', timestamp_format: 'ISO8601', label_column: 'label', auto_split: true, split_name: 'Mein Split' };
    const result = await timeSeriesApi.createDataset(file, metadata);
    expect(result.split_count).toBe(1);
    expect(request).toHaveBeenCalledTimes(1);
    const [url, options] = request.mock.calls[0];
    expect(url).toBe('/api/time-series/datasets');
    expect(JSON.parse((options!.body as FormData).get('metadata') as string)).toEqual(metadata);
  });
});
