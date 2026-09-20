import { afterEach, describe, expect, it, vi } from 'vitest';
import { pipelineBlockers, watchPipelinePreview, type PreviewState } from './pipelinePreview';
import type { PipelinePreview } from './types';
const preview = { errors: [], window_length: 36 } as unknown as PipelinePreview;
const complete = { name: 'Test', datasetSelected: true, splitSelected: true, modelSelected: true };
afterEach(() => vi.useRealTimers());
describe('pipeline readiness', () => {
  it('explains missing selections, pending requests and server errors', () => {
    expect(pipelineBlockers({ name: '', datasetSelected: false, splitSelected: false, modelSelected: false })).toHaveLength(4);
    expect(pipelineBlockers(complete)[0]).toContain('geprüft');
    expect(pipelineBlockers({ ...complete, state: { key: 'x', status: 'error', error: 'Server nicht erreichbar' } })).toEqual(['Server nicht erreichbar']);
  });
  it('enables actions only for a successful complete preview', () => {
    expect(pipelineBlockers({ ...complete, state: { key: 'x', status: 'ready', preview } })).toEqual([]);
    expect(pipelineBlockers({ ...complete, state: { key: 'x', status: 'ready', preview: { ...preview, errors: ['test: kein vollständiges Fenster'] } } })).toEqual(['test: kein vollständiges Fenster']);
    expect(pipelineBlockers({ ...complete, state: { key: 'x', status: 'ready' } })[0]).toContain('fehlt');
  });
});
describe('preview lifecycle', () => {
  it('reports loading followed by ready after debounce', async () => {
    vi.useFakeTimers();
    const update = vi.fn();
    const request = vi.fn().mockResolvedValue(preview);
    const stop = watchPipelinePreview('current', request, update);
    expect(update).toHaveBeenLastCalledWith({ key: 'current', status: 'loading' });
    expect(request).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(300);
    expect(update).toHaveBeenLastCalledWith({ key: 'current', status: 'ready', preview });
    stop();
  });
  it('reports errors and permits a fresh retry', async () => {
    vi.useFakeTimers();
    const update = vi.fn();
    const stop = watchPipelinePreview('first', () => Promise.reject(new Error('Ungültige Sensorwerte')), update);
    await vi.advanceTimersByTimeAsync(300);
    expect(update).toHaveBeenLastCalledWith({ key: 'first', status: 'error', error: 'Ungültige Sensorwerte' });
    stop();
    const retry = watchPipelinePreview('retry', async () => preview, update);
    await vi.advanceTimersByTimeAsync(300);
    expect(update).toHaveBeenLastCalledWith({ key: 'retry', status: 'ready', preview });
    retry();
  });
  it('aborts stalled requests and ignores late responses', async () => {
    vi.useFakeTimers();
    let finish!: (value: PipelinePreview) => void;
    let signal!: AbortSignal;
    const states: PreviewState[] = [];
    const stop = watchPipelinePreview('slow', s => { signal = s; return new Promise(resolve => { finish = resolve; }); }, state => states.push(state));
    await vi.advanceTimersByTimeAsync(60300);
    expect(signal.aborted).toBe(true);
    expect(states.at(-1)?.error).toContain('60 Sekunden');
    finish(preview);
    await vi.advanceTimersByTimeAsync(1);
    expect(states.at(-1)?.status).toBe('error');
    stop();
  });
  it('prevents obsolete selections from overwriting a newer preview', async () => {
    vi.useFakeTimers();
    let finish!: (value: PipelinePreview) => void;
    const update = vi.fn();
    const stop = watchPipelinePreview('old', () => new Promise(resolve => { finish = resolve; }), update);
    await vi.advanceTimersByTimeAsync(300);
    stop();
    const next = watchPipelinePreview('new', async () => preview, update);
    await vi.advanceTimersByTimeAsync(300);
    finish({ ...preview, errors: ['obsolete'] });
    await vi.advanceTimersByTimeAsync(1);
    expect(update).toHaveBeenLastCalledWith({ key: 'new', status: 'ready', preview });
    next();
  });
});
