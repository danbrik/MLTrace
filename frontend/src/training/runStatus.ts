import type { TrainingRunStatus } from '../types';

export function runStatusColor(status: TrainingRunStatus | string): string {
  if (status === 'finished') return 'green';
  if (status === 'running') return 'blue';
  if (status === 'aborted') return 'orange';
  if (status === 'failed') return 'red';
  return 'gray'; // queued
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${(seconds / 3600).toFixed(2)}h`;
}

export function formatLoss(value: number | null): string {
  return value === null || value === undefined ? '—' : value.toFixed(4);
}
