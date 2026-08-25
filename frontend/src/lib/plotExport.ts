import type { Data } from './plotly';
import {
  normalizedTableDownloadName,
  tabularTableToCsv,
  tabularTableToParquet,
  type TabularValue,
} from './tabularExport';

export type PlotExportValue = TabularValue;

export type PlotExportColumn = {
  name: string;
  values: PlotExportValue[];
};

export type PlotExportTable = {
  columns: PlotExportColumn[];
  rowCount: number;
  seriesCount: number;
};

type RawSeries = {
  name: string;
  points: Array<{ x: unknown; value: unknown }>;
};

type AxisKind = 'timestamp' | 'number' | 'category';

const TIMESTAMP_PATTERN = /^(\d{4})[-.](\d{2})[-.](\d{2})(?:[T\s](\d{2}):(\d{2})(?::(\d{2}))?)?/;

function pad(value: number): string {
  return String(value).padStart(2, '0');
}

/** Format timestamps without changing MLTrace's dataset-local wall-clock time. */
export function formatPlotTimestamp(value: unknown): string | null {
  if (value instanceof Date && Number.isFinite(value.getTime())) {
    return `${value.getFullYear()}.${pad(value.getMonth() + 1)}.${pad(value.getDate())} ${pad(value.getHours())}:${pad(value.getMinutes())}:${pad(value.getSeconds())}`;
  }
  if (typeof value !== 'string') return null;
  const match = value.trim().match(TIMESTAMP_PATTERN);
  if (!match) return null;
  return `${match[1]}.${match[2]}.${match[3]} ${match[4] ?? '00'}:${match[5] ?? '00'}:${match[6] ?? '00'}`;
}

function exportValue(value: unknown): PlotExportValue {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' || typeof value === 'boolean') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function traceName(trace: Record<string, unknown>, index: number): string {
  const name = typeof trace.name === 'string' ? trace.name.trim() : '';
  return name || `series_${index + 1}`;
}

function uniqueNames(series: RawSeries[]): RawSeries[] {
  const counts = new Map<string, number>();
  return series.map((item) => {
    const count = (counts.get(item.name) ?? 0) + 1;
    counts.set(item.name, count);
    return { ...item, name: count === 1 ? item.name : `${item.name}_${count}` };
  });
}

function vectorSeries(trace: Record<string, unknown>, index: number): RawSeries[] {
  const y = trace.y;
  if (!Array.isArray(y)) return [];
  const x = Array.isArray(trace.x) ? trace.x : y.map((_, pointIndex) => pointIndex);
  const length = Math.min(x.length, y.length);
  return [{
    name: traceName(trace, index),
    points: Array.from({ length }, (_, pointIndex) => ({ x: x[pointIndex], value: y[pointIndex] })),
  }];
}

function heatmapSeries(trace: Record<string, unknown>, index: number): RawSeries[] {
  if (trace.type !== 'heatmap' || !Array.isArray(trace.z)) return [];
  const baseName = traceName(trace, index);
  const yLabels = Array.isArray(trace.y) ? trace.y : [];
  const xValues = Array.isArray(trace.x) ? trace.x : [];
  return trace.z.flatMap((row, rowIndex) => {
    if (!Array.isArray(row)) return [];
    const label = yLabels[rowIndex] ?? rowIndex;
    return [{
      name: `${baseName} · ${String(label)}`,
      points: row.map((value, columnIndex) => ({ x: xValues[columnIndex] ?? columnIndex, value })),
    }];
  });
}

function rawSeries(data: Data[]): RawSeries[] {
  return uniqueNames(data.flatMap((input, index) => {
    const trace = input as unknown as Record<string, unknown>;
    const heatmap = heatmapSeries(trace, index);
    return heatmap.length > 0 ? heatmap : vectorSeries(trace, index);
  }).filter((series) => series.points.length > 0));
}

function axisKind(series: RawSeries[]): AxisKind {
  const values = series.flatMap((item) => item.points.map((point) => point.x));
  if (values.length > 0 && values.every((value) => formatPlotTimestamp(value) !== null)) return 'timestamp';
  if (values.length > 0 && values.every((value) => typeof value === 'number' && Number.isFinite(value))) return 'number';
  return 'category';
}

function normalizedAxisValue(value: unknown, kind: AxisKind): string | number {
  if (kind === 'timestamp') return formatPlotTimestamp(value) ?? String(value);
  if (kind === 'number') return value as number;
  if (value === null || value === undefined) return '';
  return String(value);
}

function axisKey(value: string | number): string {
  return `${typeof value}:${String(value)}`;
}

/**
 * Convert every Plotly trace into one outer-joined table. Duplicate X values
 * are retained by matching their first/second/etc. occurrence across traces.
 */
export function buildPlotExportTable(data: Data[]): PlotExportTable {
  const series = rawSeries(data);
  if (series.length === 0) return { columns: [], rowCount: 0, seriesCount: 0 };
  const kind = axisKind(series);
  const rowAxes = new Map<string, { value: string | number; occurrence: number; order: number }>();
  const pointMaps: Array<Map<string, PlotExportValue>> = [];
  let order = 0;

  series.forEach((item) => {
    const occurrences = new Map<string, number>();
    const points = new Map<string, PlotExportValue>();
    item.points.forEach((point) => {
      const value = normalizedAxisValue(point.x, kind);
      const baseKey = axisKey(value);
      const occurrence = occurrences.get(baseKey) ?? 0;
      occurrences.set(baseKey, occurrence + 1);
      const key = `${baseKey}\u0000${occurrence}`;
      if (!rowAxes.has(key)) rowAxes.set(key, { value, occurrence, order: order++ });
      points.set(key, exportValue(point.value));
    });
    pointMaps.push(points);
  });

  const rows = [...rowAxes.entries()].sort(([, left], [, right]) => {
    if (kind === 'category') return left.order - right.order;
    if (left.value < right.value) return -1;
    if (left.value > right.value) return 1;
    return left.occurrence - right.occurrence;
  });
  const columns: PlotExportColumn[] = [{
    name: kind === 'timestamp' ? 'timestamp' : 'x',
    values: rows.map(([, row]) => row.value),
  }];
  series.forEach((item, index) => {
    columns.push({ name: item.name, values: rows.map(([key]) => pointMaps[index].get(key) ?? null) });
  });
  return { columns, rowCount: rows.length, seriesCount: series.length };
}

export async function resolvePlotExportTable(
  data: Data[],
  fullResolutionExport?: () => Promise<PlotExportTable>,
): Promise<PlotExportTable> {
  return fullResolutionExport ? fullResolutionExport() : buildPlotExportTable(data);
}

export function plotTableToCsv(table: PlotExportTable): string {
  return tabularTableToCsv(table);
}

export async function plotTableToParquet(table: PlotExportTable): Promise<ArrayBuffer> {
  return tabularTableToParquet(table, 'MLTrace plot export');
}

export function normalizedDownloadName(input: string, extension: 'csv' | 'parquet'): string {
  return normalizedTableDownloadName(input, extension, 'plot-data');
}
