import type { TimeRangeValue } from './types';

// Compare dataset wall-clock values directly, including optional microseconds.
function localKey(value: string): string | null {
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?$/.exec(value);
  if (!match) return null;
  const [, date, hour, minute, second = '00', fraction = ''] = match;
  if (+hour > 23 || +minute > 59 || +second > 59) return null;
  const parsed = new Date(`${date}T00:00:00Z`);
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== date) return null;
  return `${date}T${hour}:${minute}:${second}.${fraction.padEnd(6, '0')}`;
}

export function rangeProblem(range: TimeRangeValue, min?: string, max?: string): string | null {
  const start = localKey(range.start), end = localKey(range.end);
  if (!start || !end) return 'Bitte gültigen Beginn und gültiges Ende angeben.';
  if (start > end) return 'Beginn darf nicht nach dem Ende liegen.';
  const lower = min ? localKey(min) : null, upper = max ? localKey(max) : null;
  if ((lower && start < lower) || (upper && end > upper)) return 'Außerhalb des verfügbaren Datensatzzeitraums.';
  return null;
}

export const formatPresetTime = (value: string) => value.replace('T', ' ');

/** Copy only timestamps; callers retain their own sampling and other settings. */
export function copyTimeRange<T extends TimeRangeValue>(current: T, source: TimeRangeValue): T {
  return { ...current, start: source.start, end: source.end };
}
