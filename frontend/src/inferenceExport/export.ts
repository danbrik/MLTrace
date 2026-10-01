import type { PlotExportTable } from '../lib/plotExport';
import type { TestingRunPlotSeriesPoint } from '../types';

export type SamplingRate = 'original' | '1min' | '5min';
export type SamplingMethod = 'first' | 'mean' | 'median' | 'min' | 'max';
export type ExportOptions = { samplingRate: SamplingRate; method: SamplingMethod; utc: boolean };
export type InferenceExportSource = { id: number; name: string; points: TestingRunPlotSeriesPoint[] };

const MICROSECONDS_PER_SECOND = 1_000_000n;
const berlinOffsetFormatter = new Intl.DateTimeFormat('en-US', {
  timeZone: 'Europe/Berlin', timeZoneName: 'longOffset',
});

function offsetAt(milliseconds: number): number {
  const offset = berlinOffsetFormatter.formatToParts(milliseconds).find((part) => part.type === 'timeZoneName')?.value;
  if (offset === 'GMT') return 0;
  const match = offset?.match(/^GMT([+-])(\d{2}):(\d{2})(?::(\d{2}))?$/);
  if (!match) throw new Error('Could not determine the Europe/Berlin time zone offset.');
  return (match[1] === '+' ? 1 : -1) * (Number(match[2]) * 3600 + Number(match[3]) * 60 + Number(match[4] ?? 0)) * 1000;
}

function floorDivide(value: bigint, divisor: bigint): bigint {
  const quotient = value / divisor;
  return value < 0n && value % divisor !== 0n ? quotient - 1n : quotient;
}

/** Keep source microseconds intact until joining; never interpret dates in the browser's zone. */
function parseTimestamp(timestamp: string, utc: boolean, offsetsByDay: Map<string, number[]>): bigint {
  const match = timestamp.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d{1,6}))?$/);
  if (!match) throw new Error(`Invalid source timestamp: ${timestamp}`);
  const canonical = `${match[1]}T${match[2]}`;
  const wallClock = Date.parse(`${canonical}Z`);
  if (!Number.isFinite(wallClock) || new Date(wallClock).toISOString().slice(0, 19) !== canonical) {
    throw new Error(`Invalid source timestamp: ${timestamp}`);
  }
  const fraction = BigInt((match[3] ?? '').padEnd(6, '0'));
  if (!utc) return BigInt(wallClock) * 1000n + fraction;

  let offsets = offsetsByDay.get(match[1]);
  if (!offsets) {
    const midnight = Date.parse(`${match[1]}T00:00:00Z`);
    // Adjacent days include both possible offsets on Berlin's DST transition days.
    offsets = [...new Set([-36, 0, 36].map((hours) => offsetAt(midnight + hours * 3600_000)))];
    offsetsByDay.set(match[1], offsets);
  }
  const candidates = offsets.map((offset) => wallClock - offset)
    .filter((candidate, index) => offsets.length === 1 || offsetAt(candidate) === offsets[index]);
  if (candidates.length !== 1) {
    throw new Error(`${candidates.length ? 'Ambiguous' : 'Nonexistent'} Europe/Berlin local time: ${timestamp}. UTC export cannot determine a unique instant at this daylight-saving transition.`);
  }
  return BigInt(candidates[0]) * 1000n + fraction;
}

function formatTimestamp(microseconds: bigint): string {
  const milliseconds = Number(floorDivide(microseconds, MICROSECONDS_PER_SECOND)) * 1000;
  return new Date(milliseconds).toISOString().slice(0, 19).replaceAll('-', '.').replace('T', ' ');
}

function aggregate(values: number[], method: SamplingMethod): number {
  switch (method) {
    case 'first': return values[0];
    case 'mean': return values.reduce((sum, value) => sum + value / values.length, 0);
    case 'min': return values.reduce((minimum, value) => Math.min(minimum, value));
    case 'max': return values.reduce((maximum, value) => Math.max(maximum, value));
    case 'median': {
      values.sort((a, b) => a - b);
      const middle = Math.floor(values.length / 2);
      return values.length % 2 ? values[middle] : values[middle - 1] / 2 + values[middle] / 2;
    }
  }
}

export function buildInferenceExport(sources: InferenceExportSource[], options: ExportOptions): PlotExportTable {
  if (!sources.length) throw new Error('Select at least one finished inference.');
  if (new Set(sources.map((source) => source.id)).size !== sources.length) throw new Error('An inference was selected more than once.');
  const offsetsByDay = new Map<string, number[]>();
  const axes = new Map<string, { timestamp: bigint; occurrence: number }>();
  const interval = options.samplingRate === 'original' ? null : BigInt(options.samplingRate === '1min' ? 60 : 300) * MICROSECONDS_PER_SECOND;
  const series = sources.map((source) => {
    if (!source.points.length) throw new Error(`Inference "${source.name}" (#${source.id}) has no stored points.`);
    try {
      const sorted = source.points.map((point) => {
        if (!Number.isFinite(point.value)) throw new Error(`Missing or non-finite score at ${point.timestamp}.`);
        return { ...point, time: parseTimestamp(point.timestamp, options.utc, offsetsByDay) };
      }).sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1 : a.position - b.position);
      const samples: Array<{ time: bigint; value: number }> = [];
      if (interval !== null) {
        let bucket: bigint | null = null;
        let values: number[] = [];
        const flush = () => {
          if (bucket === null) return;
          const value = aggregate(values, options.method);
          if (!Number.isFinite(value)) throw new Error('Aggregation produced a non-finite score.');
          samples.push({ time: bucket, value });
        };
        for (const point of sorted) {
          const nextBucket = floorDivide(point.time, interval) * interval;
          if (nextBucket !== bucket) {
            flush();
            bucket = nextBucket;
            values = [];
          }
          values.push(point.value);
        }
        flush();
      }
      const occurrences = new Map<bigint, number>();
      const values = new Map<string, number>();
      for (const point of interval === null ? sorted : samples) {
        const occurrence = occurrences.get(point.time) ?? 0;
        occurrences.set(point.time, occurrence + 1);
        const key = `${point.time}:${occurrence}`;
        axes.set(key, { timestamp: point.time, occurrence });
        values.set(key, point.value);
      }
      return values;
    } catch (error) {
      throw new Error(`Inference "${source.name}" (#${source.id}): ${error instanceof Error ? error.message : String(error)}`);
    }
  });
  const rows = [...axes.entries()].sort(([, a], [, b]) => a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : a.occurrence - b.occurrence);
  return {
    columns: [
      { name: 'timestamp', values: rows.map(([, row]) => formatTimestamp(row.timestamp)) },
      ...sources.map((source, index) => ({ name: `${source.name} (#${source.id})`, values: rows.map(([key]) => series[index].get(key) ?? null) })),
    ],
    rowCount: rows.length,
    seriesCount: sources.length,
  };
}
