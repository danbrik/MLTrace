import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { requestRoiPreview } from './roiPreview';

describe('ROI heatmap preview requests', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {vi.useRealTimers(); vi.unstubAllGlobals();});

  it('debounces rapid slider changes by 150 ms', async () => {
    const fetcher = vi.fn().mockResolvedValue({ok: true, blob: async () => new Blob(['image'])});
    vi.stubGlobal('fetch', fetcher);
    const ready = vi.fn(), failed = vi.fn();
    const cancel = requestRoiPreview('/old', ready, failed);
    await vi.advanceTimersByTimeAsync(100);
    cancel();
    const stop = requestRoiPreview('/new', ready, failed);
    await vi.advanceTimersByTimeAsync(149);
    expect(fetcher).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0][0]).toBe('/new');
    expect(ready).toHaveBeenCalledTimes(1);
    expect(failed).not.toHaveBeenCalled();
    stop();
  });

  it('ignores late responses after a pair, run, or project switch even if abort is ignored', async () => {
    let finish!: (value: Response) => void;
    const fetcher = vi.fn().mockReturnValue(new Promise<Response>(resolve => {finish = resolve;}));
    vi.stubGlobal('fetch', fetcher);
    const ready = vi.fn(), failed = vi.fn();
    const cancel = requestRoiPreview('/previous-project', ready, failed);
    await vi.advanceTimersByTimeAsync(150);
    cancel();
    expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true);
    finish({ok: true, blob: async () => new Blob(['stale'])} as Response);
    await vi.runAllTimersAsync();
    expect(ready).not.toHaveBeenCalled();
    expect(failed).not.toHaveBeenCalled();
  });

  it('reports failed requests and suppresses failures after cancellation', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', {status: 409})));
    const ready = vi.fn(), failed = vi.fn();
    const cancel = requestRoiPreview('/missing', ready, failed);
    await vi.advanceTimersByTimeAsync(150);
    expect(failed).toHaveBeenCalledWith(expect.stringContaining('nicht geladen'));
    expect(ready).not.toHaveBeenCalled();
    cancel();
    const stopped = requestRoiPreview('/cancelled', ready, failed);
    stopped();
    await vi.runAllTimersAsync();
    expect(failed).toHaveBeenCalledTimes(1);
  });
});
