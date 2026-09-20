import type { TimeInterval } from './types';

// Preserve sub-millisecond CSV timestamps when checking inclusive boundaries.
export function timestampKey(value: string): bigint | null {
  const match = value.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z|[+-]\d{2}:\d{2})?$/);
  if (!match) return null;
  const [year, month, day] = match[1].split('-').map(Number);
  const [hour, minute] = match[2].split(':').map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const daysInMonth = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (month < 1 || month > 12 || day < 1 || day > daysInMonth[month - 1] || hour > 23 || minute > 59 || Number(match[3] ?? 0) > 59) return null;
  const seconds = `${match[1]}T${match[2]}:${match[3] ?? '00'}${match[5] ?? 'Z'}`;
  const millis = Date.parse(seconds);
  if (!Number.isFinite(millis)) return null;
  return BigInt(millis) * 1000000n + BigInt((match[4] ?? '').padEnd(9, '0'));
}

export function intervalError(candidate: TimeInterval, intervals: TimeInterval[], bounds: { start: string; end: string }): string | null {
  const start = timestampKey(candidate.start), end = timestampKey(candidate.end);
  if (start === null || end === null) return 'Bitte Beginn und Ende vollständig eingeben.';
  if (start > end) return 'Der Beginn darf nicht nach dem Ende liegen.';
  if (start < timestampKey(bounds.start)! || end > timestampKey(bounds.end)!) return 'Der Zeitraum muss innerhalb der Datenbasis liegen.';
  const conflict = intervals.find((item) => item.id !== candidate.id && start <= timestampKey(item.end)! && end >= timestampKey(item.start)!);
  if (conflict) return `Dieser Zeitraum ist bereits teilweise oder vollständig ${conflict.subset === 'validation' ? 'Validation' : conflict.subset === 'train' ? 'Train' : 'Test'} zugeordnet (${displayTime(conflict.start)} – ${displayTime(conflict.end)}).`;
  return null;
}

export function sortIntervals(intervals: TimeInterval[]): TimeInterval[] {
  return [...intervals].sort((a, b) => timestampKey(a.start)! < timestampKey(b.start)! ? -1 : timestampKey(a.start)! > timestampKey(b.start)! ? 1 : 0);
}

export function displayTime(value: string): string { return value.replace('T', ' ').replace(/Z$/, ''); }
export function inputTime(value: string): string { return value.replace(/Z$/, ''); }
export function utcTime(value: string): string { return /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`; }
