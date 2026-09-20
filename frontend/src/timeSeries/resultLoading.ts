/** Ignore superseded responses and make stalled requests recoverable. */
export function loadResults<T>(request: (signal: AbortSignal) => Promise<T>, callbacks: {
  success: (value: T) => void; error: (message: string) => void; settled: () => void;
}) {
  const controller = new AbortController();
  let stopped = false;
  const timeout = setTimeout(() => {
    if (stopped) return;
    stopped = true;
    controller.abort();
    callbacks.error('Das Laden hat nach 60 Sekunden nicht geantwortet. Bitte erneut laden.');
    callbacks.settled();
  }, 60000);
  request(controller.signal).then(value => {
    if (!stopped) callbacks.success(value);
  }).catch(error => {
    if (!stopped) callbacks.error(error instanceof Error ? error.message : String(error));
  }).finally(() => {
    clearTimeout(timeout);
    if (!stopped) callbacks.settled();
  });
  return () => { stopped = true; clearTimeout(timeout); controller.abort(); };
}
