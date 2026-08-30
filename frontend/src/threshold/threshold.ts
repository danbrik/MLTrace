import type { CsvCell, CsvDocument } from '../csvMerge/csvMerge';
import { parseCsvTimestamp } from '../csvMerge/csvMerge';
import { formatPlotTimestamp, type PlotExportTable } from '../lib/plotExport';

export type ThresholdScope = 'entire' | 'reference';

export type ThresholdRow = {
  timestamp: string;
  originalTimestamp: string;
  position: number;
  continuitySegment: number;
  values: Record<string, number | null>;
};

export type ThresholdDataset = {
  rows: ThresholdRow[];
  invalidTimestampRows: number;
  invalidNumericByColumn: Record<string, number>;
};

export type QuantileThresholdResult = {
  value: number;
  inputCount: number;
  exceedancePointCount: number;
  intervals: ThresholdInterval[];
};

export type ThresholdInterval = {
  start: string;
  end: string;
  continuitySegment: number;
};

export type CalculatedThreshold = {
  id: string;
  column: string;
  quantile: number;
  value: number;
};

export const THRESHOLD_COLORS = ['#228be6', '#f08c00', '#2f9e44', '#ae3ec9', '#0c8599', '#e03131'];

export type ThresholdAxisPlan = {
  column: string;
  color: string;
  traceAxis: string;
  layoutAxis: string;
  side: 'left' | 'right';
  position: number;
};

export function thresholdAxisPlan(columns: string[]): { axes: ThresholdAxisPlan[]; xDomain: [number, number] } {
  if (columns.length > 6) throw new Error('At most six graph columns can be displayed.');
  const leftCount = Math.ceil(columns.length / 2);
  const rightCount = Math.floor(columns.length / 2);
  let leftIndex = 0;
  let rightIndex = 0;
  const axes = columns.map((column, index) => {
    const side = index % 2 === 0 ? 'left' as const : 'right' as const;
    const localIndex = side === 'left' ? leftIndex++ : rightIndex++;
    const sideCount = side === 'left' ? leftCount : rightCount;
    const position = side === 'left' ? (sideCount - localIndex) * 0.055 : 1 - (sideCount - localIndex) * 0.055;
    return {
      column,
      color: THRESHOLD_COLORS[index],
      traceAxis: index === 0 ? 'y' : `y${index + 1}`,
      layoutAxis: index === 0 ? 'yaxis' : `yaxis${index + 1}`,
      side,
      position,
    };
  });
  return { axes, xDomain: [leftCount * 0.055 + 0.025, 1 - rightCount * 0.055 - 0.025] };
}

const MINIMUM_GAP_MS = 15_000;
const GAP_FACTOR = 5;

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function timestampMillis(value: string): number {
  return new Date(`${value.replace(/Z$/, '')}Z`).getTime();
}

function datasetLocalIso(milliseconds: number): string {
  const date = new Date(milliseconds);
  const pad = (value: number, width = 2) => String(value).padStart(width, '0');
  const fraction = date.getUTCMilliseconds() ? `.${pad(date.getUTCMilliseconds(), 3).replace(/0+$/, '')}` : '';
  return `${pad(date.getUTCFullYear(), 4)}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}T${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}${fraction}`;
}

export function parseThresholdNumber(value: CsvCell): { value: number | null; invalid: boolean } {
  if (value === null || value.trim() === '') return { value: null, invalid: false };
  let text = value.trim();
  if (text.includes(',') && !text.includes('.') && text.split(',').length === 2) text = text.replace(',', '.');
  const parsed = Number(text);
  return Number.isFinite(parsed) ? { value: parsed, invalid: false } : { value: null, invalid: true };
}

function selectionTimestamp(value: string): string {
  const match = value.match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2})?)/);
  return match ? `${match[1]}T${match[2].length === 5 ? `${match[2]}:00` : match[2]}` : value.slice(0, 19).replace(' ', 'T');
}

export function canonicalSelectionRange(left: string, right: string): { start: string; end: string } {
  const first = selectionTimestamp(left);
  const second = selectionTimestamp(right);
  return first <= second ? { start: first, end: second } : { start: second, end: first };
}

export function timestampCandidateColumns(document: CsvDocument): string[] {
  return document.headers.filter((_, columnIndex) => {
    const populated = document.rows.map((row) => row[columnIndex]).filter((value): value is string => value !== null && value.trim() !== '');
    if (populated.length === 0) return false;
    return populated.some((value) => parseCsvTimestamp(value) !== null);
  });
}

export function numericCandidateColumns(document: CsvDocument, excluded?: string | null): string[] {
  return document.headers.filter((header, columnIndex) => {
    if (header === excluded) return false;
    const populated = document.rows.map((row) => row[columnIndex]).filter((value): value is string => value !== null && value.trim() !== '');
    if (populated.length === 0) return false;
    return populated.some((value) => parseThresholdNumber(value).value !== null);
  });
}

function continuitySegments(timestamps: string[]): number[] {
  if (timestamps.length === 0) return [];
  const deltas = timestamps.slice(1)
    .map((timestamp, index) => timestampMillis(timestamp) - timestampMillis(timestamps[index]))
    .filter((value) => value > 0);
  const cadence = median(deltas);
  const threshold = Math.max(MINIMUM_GAP_MS, (cadence ?? 0) * GAP_FACTOR);
  const segments = [0];
  let segment = 0;
  for (let index = 1; index < timestamps.length; index += 1) {
    if (timestampMillis(timestamps[index]) - timestampMillis(timestamps[index - 1]) > threshold) segment += 1;
    segments.push(segment);
  }
  return segments;
}

export function buildThresholdDataset(document: CsvDocument, timeColumn: string, columns: string[]): ThresholdDataset {
  const timeIndex = document.headers.indexOf(timeColumn);
  if (timeIndex < 0) throw new Error(`Time column '${timeColumn}' does not exist.`);
  const columnIndexes = Object.fromEntries(columns.map((column) => [column, document.headers.indexOf(column)]));
  const invalidNumericByColumn = Object.fromEntries(columns.map((column) => [column, 0]));
  let invalidTimestampRows = 0;
  const rows: Omit<ThresholdRow, 'continuitySegment'>[] = [];
  document.rows.forEach((sourceRow, position) => {
    const originalTimestamp = sourceRow[timeIndex];
    if (originalTimestamp === null) { invalidTimestampRows += 1; return; }
    const parsedTimestamp = parseCsvTimestamp(originalTimestamp);
    if (!parsedTimestamp) { invalidTimestampRows += 1; return; }
    const values: Record<string, number | null> = {};
    columns.forEach((column) => {
      const parsed = parseThresholdNumber(sourceRow[columnIndexes[column]] ?? null);
      values[column] = parsed.value;
      if (parsed.invalid) invalidNumericByColumn[column] += 1;
    });
    rows.push({ timestamp: parsedTimestamp.canonical, originalTimestamp, position, values });
  });
  rows.sort((left, right) => left.timestamp.localeCompare(right.timestamp) || left.position - right.position);
  const segments = continuitySegments(rows.map((row) => row.timestamp));
  return {
    rows: rows.map((row, index) => ({ ...row, continuitySegment: segments[index] })),
    invalidTimestampRows,
    invalidNumericByColumn,
  };
}

export function linearQuantile(values: number[], quantile: number): number {
  if (!Number.isFinite(quantile) || quantile < 0 || quantile > 1) throw new Error('Quantile must be between 0 and 1.');
  const finite = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (finite.length === 0) throw new Error('The selected quantile source contains no finite values.');
  if (finite.length === 1) return finite[0];
  const index = (finite.length - 1) * quantile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return finite[lower] * (1 - weight) + finite[upper] * weight;
}

function intervalEnd(rows: ThresholdRow[], lastIndex: number): string {
  const current = rows[lastIndex];
  const next = rows[lastIndex + 1];
  if (next && next.continuitySegment === current.continuitySegment) return next.timestamp;
  const previous = rows[lastIndex - 1];
  if (previous && previous.continuitySegment === current.continuitySegment) {
    const delta = timestampMillis(current.timestamp) - timestampMillis(previous.timestamp);
    if (delta > 0) return datasetLocalIso(timestampMillis(current.timestamp) + delta);
  }
  return current.timestamp;
}

export function calculateQuantileThreshold(
  rows: ThresholdRow[],
  column: string,
  quantile: number,
  scope: ThresholdScope,
  referenceStart?: string,
  referenceEnd?: string,
): QuantileThresholdResult {
  let sourceRows = rows;
  if (scope === 'reference') {
    if (!referenceStart || !referenceEnd) throw new Error('Reference start and end are required.');
    const start = referenceStart <= referenceEnd ? referenceStart : referenceEnd;
    const end = referenceStart <= referenceEnd ? referenceEnd : referenceStart;
    sourceRows = rows.filter((row) => row.timestamp >= start && row.timestamp <= end);
  }
  const sourceValues = sourceRows.flatMap((row) => {
    const value = row.values[column];
    return value === null || !Number.isFinite(value) ? [] : [value];
  });
  const value = linearQuantile(sourceValues, quantile);
  const above = rows.map((row) => row.values[column] !== null && Number(row.values[column]) > value);
  const intervals: ThresholdInterval[] = [];
  let index = 0;
  while (index < rows.length) {
    if (!above[index]) { index += 1; continue; }
    const startIndex = index;
    const segment = rows[index].continuitySegment;
    while (index + 1 < rows.length && above[index + 1] && rows[index + 1].continuitySegment === segment) index += 1;
    intervals.push({ start: rows[startIndex].timestamp, end: intervalEnd(rows, index), continuitySegment: segment });
    index += 1;
  }
  return {
    value,
    inputCount: sourceValues.length,
    exceedancePointCount: above.filter(Boolean).length,
    intervals,
  };
}

export function unionThresholdIntervals(intervals: ThresholdInterval[]): ThresholdInterval[] {
  const sorted = [...intervals].sort((left, right) => left.continuitySegment - right.continuitySegment || left.start.localeCompare(right.start));
  const merged: ThresholdInterval[] = [];
  sorted.forEach((interval) => {
    const current = merged.at(-1);
    if (current && current.continuitySegment === interval.continuitySegment && interval.start <= current.end) {
      if (interval.end > current.end) current.end = interval.end;
    } else {
      merged.push({ ...interval });
    }
  });
  return merged;
}

export function decimateThresholdRows(rows: ThresholdRow[], columns: string[], maximum = 8000): ThresholdRow[] {
  if (rows.length <= maximum || maximum <= 0) return rows;
  const bucketCount = Math.max(1, Math.floor(maximum / Math.max(2, columns.length * 2)));
  const width = Math.max(1, Math.ceil(rows.length / bucketCount));
  const selected = new Set<number>([0, rows.length - 1]);
  for (let start = 0; start < rows.length; start += width) {
    const end = Math.min(rows.length, start + width);
    selected.add(start); selected.add(end - 1);
    columns.forEach((column) => {
      const finite = Array.from({ length: end - start }, (_, offset) => start + offset)
        .filter((index) => rows[index].values[column] !== null);
      if (finite.length > 0) {
        selected.add(finite.reduce((best, index) => Number(rows[index].values[column]) < Number(rows[best].values[column]) ? index : best));
        selected.add(finite.reduce((best, index) => Number(rows[index].values[column]) > Number(rows[best].values[column]) ? index : best));
      }
    });
  }
  for (let index = 1; index < rows.length; index += 1) {
    if (rows[index].continuitySegment !== rows[index - 1].continuitySegment) { selected.add(index - 1); selected.add(index); }
  }
  const ordered = [...selected].sort((left, right) => left - right);
  if (ordered.length <= maximum) return ordered.map((index) => rows[index]);
  const stride = ordered.length / maximum;
  return Array.from({ length: maximum }, (_, index) => rows[ordered[Math.min(ordered.length - 1, Math.floor(index * stride))]]);
}

export function thresholdExportTable(
  rows: ThresholdRow[],
  columns: string[],
  thresholds: CalculatedThreshold[],
): PlotExportTable {
  return {
    columns: [
      { name: 'timestamp', values: rows.map((row) => formatPlotTimestamp(row.timestamp) ?? row.timestamp) },
      ...columns.map((column) => ({ name: column, values: rows.map((row) => row.values[column]) })),
      ...thresholds.map((threshold) => ({
        name: `Threshold · ${threshold.column} · q=${threshold.quantile} · ${threshold.id.slice(-6)}`,
        values: rows.map(() => threshold.value),
      })),
    ],
    rowCount: rows.length,
    seriesCount: columns.length + thresholds.length,
  };
}
