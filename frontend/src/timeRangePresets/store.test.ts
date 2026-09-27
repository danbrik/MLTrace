import { describe, expect, it, vi } from 'vitest';
import { createTimeRangePresetStore } from './store';
import type { TimeRangePreset, TimeRangePresetInput } from './types';
const input: TimeRangePresetInput = { name: 'Normal', start: '2026-01-01T10:00:00', end: '2026-01-01T11:00:00' };
const row: TimeRangePreset = { ...input, id: 1, created_at: 'now', updated_at: 'now' };
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
};
function client() {
  return { list: vi.fn(async (_project: string) => [row]), create: vi.fn(async (_project: string, value: TimeRangePresetInput) => ({ ...row, ...value })),
    update: vi.fn(async (_project: string, id: number, value: TimeRangePresetInput) => ({ ...row, ...value, id })), remove: vi.fn(async (_project: string, _id: number) => {}) };
}
describe('shared project preset store', () => {
  it('synchronizes both selectors after create, edit and delete', async () => {
    const api = client(), store = createTimeRangePresetStore(api);
    const reference = vi.fn(), anomaly = vi.fn();
    store.subscribe('one', reference); const unsubscribe = store.subscribe('one', anomaly);
    await store.create('one', input);
    expect(reference).toHaveBeenCalledTimes(1); expect(anomaly).toHaveBeenCalledTimes(1);
    await store.update('one', 1, { ...input, name: 'Changed' });
    expect(store.snapshot('one').items[0].name).toBe('Changed');
    await store.remove('one', 1);
    expect(store.snapshot('one').items).toEqual([]);
    expect(reference).toHaveBeenCalledTimes(3); expect(anomaly).toHaveBeenCalledTimes(3);
    unsubscribe();
    await store.create('one', input);
    expect(anomaly).toHaveBeenCalledTimes(3);
  });
  it('isolates projects and pending results after a project switch', async () => {
    const api = client(), pending = deferred<TimeRangePreset[]>();
    api.list.mockReturnValueOnce(pending.promise);
    const store = createTimeRangePresetStore(api);
    const oldLoad = store.load('old');
    expect(store.snapshot('new').items).toEqual([]);
    await store.create('new', { ...input, name: 'New project' });
    pending.resolve([row]); await oldLoad;
    expect(store.snapshot('old').items[0].name).toBe('Normal');
    expect(store.snapshot('new').items[0].name).toBe('New project');
    expect(api.create).toHaveBeenCalledWith('new', { ...input, name: 'New project' });
  });
  it('deduplicates loading and ignores stale responses after deletion', async () => {
    const api = client(), pending = deferred<TimeRangePreset[]>();
    api.list.mockReturnValueOnce(pending.promise).mockResolvedValueOnce([]);
    const store = createTimeRangePresetStore(api);
    const first = store.load('one'), second = store.load('one');
    expect(api.list).toHaveBeenCalledTimes(1);
    await store.remove('one', 1);
    pending.resolve([row]); await Promise.all([first, second]);
    expect(store.snapshot('one').items).toEqual([]);
    expect(api.list).toHaveBeenCalledTimes(2);
  });
  it('retains current entries after failed mutations and can retry loading', async () => {
    const api = client(), store = createTimeRangePresetStore(api);
    await store.load('one');
    api.update.mockRejectedValueOnce(new Error('Duplicate name'));
    await expect(store.update('one', 1, input)).rejects.toThrow('Duplicate name');
    expect(store.snapshot('one').items).toEqual([row]);
    api.list.mockRejectedValueOnce(new Error('Offline'));
    await store.load('one'); expect(store.snapshot('one').error).toBe('Offline');
    await store.load('one'); expect(store.snapshot('one').error).toBeNull();
  });
});
