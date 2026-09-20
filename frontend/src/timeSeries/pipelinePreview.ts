import type { PipelinePreview } from './types';

export type PreviewState = { key: string; status: 'loading' | 'ready' | 'error'; preview?: PipelinePreview; error?: string };

/** Debounce edits, cancel obsolete requests, and make stalled validation visible. */
export function watchPipelinePreview(
  key: string,
  request: (signal: AbortSignal) => Promise<PipelinePreview>,
  update: (state: PreviewState) => void,
) {
  let cancelled = false;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const controller = new AbortController();
  update({ key, status: 'loading' });
  const delay = setTimeout(() => {
    timeout = setTimeout(() => {
      if (cancelled) return;
      cancelled = true;
      controller.abort();
      update({ key, status: 'error', error: 'Die Prüfung hat nach 60 Sekunden nicht geantwortet. Bitte erneut prüfen; bei wiederholtem Auftreten die Verbindung zum Server prüfen.' });
    }, 60000);
    request(controller.signal).then(preview => {
      if (!cancelled) update({ key, status: 'ready', preview });
    }).catch(error => {
      if (!cancelled) update({ key, status: 'error', error: error instanceof Error ? error.message : String(error) });
    }).finally(() => clearTimeout(timeout));
  }, 300);
  return () => { cancelled = true; clearTimeout(delay); clearTimeout(timeout); controller.abort(); };
}

export function pipelineBlockers(input: {
  name: string; datasetSelected: boolean; splitSelected: boolean; modelSelected: boolean;
  state?: PreviewState;
}): string[] {
  const missing = [];
  if (!input.name.trim()) missing.push('Bitte einen Namen eingeben.');
  if (!input.datasetSelected) missing.push('Bitte eine Datenbasis aus der Liste auswählen.');
  if (!input.splitSelected) missing.push('Bitte einen gespeicherten Split dieser Datenbasis aus der Liste auswählen.');
  if (!input.modelSelected) missing.push('Bitte ein Modell aus der Liste auswählen.');
  if (missing.length) return missing;
  if (!input.state || input.state.status === 'loading') return ['Daten und Trainingskonfiguration werden geprüft. Bitte warten.'];
  if (input.state.status === 'error') return [input.state.error || 'Die Vorschau konnte nicht geprüft werden. Bitte erneut prüfen.'];
  if (!input.state.preview) return ['Die Vorschau fehlt. Bitte erneut prüfen.'];
  return input.state.preview.errors;
}
