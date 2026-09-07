import { describe, expect, it } from 'vitest';
import { mergeSpatialRunUpdate, spatialAbortState } from './abortState';
import type { SpatialSensitivityRun } from '../types';

describe('spatial cancellation state', () => {
  it('ignores stale polls and updates for a different selected run', () => {
    const pending = {id:1, status:'running', abort_requested_at:'now'} as SpatialSensitivityRun;
    const stale = {id:1, status:'running', abort_requested_at:null} as SpatialSensitivityRun;
    expect(spatialAbortState(mergeSpatialRunUpdate(pending, stale)).disabled).toBe(true);
    expect(mergeSpatialRunUpdate(pending, {...stale,id:2})).toBe(pending);
    const ended = {...pending,status:'aborted'};
    expect(mergeSpatialRunUpdate(ended, stale)).toBe(ended);
    expect(mergeSpatialRunUpdate(pending, {...stale,status:'finished'}).status).toBe('running');
  });
  it('allows queued and running cancellation', () => {
    for (const status of ['queued', 'running']) expect(spatialAbortState({status}).disabled).toBe(false);
  });
  it('restores pending cancellation from the API after reload', () => {
    const state = spatialAbortState({status:'running', abort_requested_at:'2026-09-07T12:00:00'});
    expect(state.pending).toBe(true);
    expect(state.disabled).toBe(true);
    expect(state.buttonLabel).toBe('Wird abgebrochen …');
    expect(state.message).toContain('10 Sekunden');
  });
  it('keeps waiting after SIGKILL until the process exits', () => {
    expect(spatialAbortState({status:'running', abort_requested_at:'now', force_killed_at:'later'}).pending).toBe(true);
  });
  it('distinguishes forced termination and disables terminal runs', () => {
    expect(spatialAbortState({status:'aborted', force_killed_at:'now'}).message).toContain('Zeitüberschreitung');
    for (const status of ['aborted', 'finished', 'failed']) expect(spatialAbortState({status}).disabled).toBe(true);
  });
});
