import type { SpatialSensitivityRun } from '../types';

// A poll already in flight when abort was accepted must not re-enable the button.
export function mergeSpatialRunUpdate(current: SpatialSensitivityRun, next: SpatialSensitivityRun): SpatialSensitivityRun {
  if (current.id !== next.id) return current;
  if (!['queued', 'running'].includes(current.status) && ['queued', 'running'].includes(next.status)) return current;
  const abortRequested = next.abort_requested_at ?? current.abort_requested_at;
  return { ...next, abort_requested_at: abortRequested,
    status: abortRequested && ['finished', 'failed'].includes(next.status) ? 'running' : next.status,
    force_killed_at: next.force_killed_at ?? current.force_killed_at };
}

export function spatialAbortState(run: Pick<SpatialSensitivityRun, 'status' | 'abort_requested_at' | 'force_killed_at'> | null) {
  const pending = Boolean(run?.status === 'running' && run.abort_requested_at);
  const forced = Boolean(run?.status === 'aborted' && run.force_killed_at);
  return {
    pending,
    disabled: !run || !['queued', 'running'].includes(run.status) || pending,
    buttonLabel: pending ? 'Wird abgebrochen …' : 'Analyse abbrechen',
    message: pending
      ? 'Abbruch angefordert … Reagiert der Worker nicht, wird er nach 10 Sekunden automatisch hart beendet. Der Status bleibt bis zum bestätigten Prozessende offen.'
      : forced ? 'Abgebrochen – Worker nach Zeitüberschreitung beendet.' : null,
  };
}
