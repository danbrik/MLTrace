import type { Data } from './plotly';

export type PlotGapPolicy = {
  continuity?: Array<number | string | null>;
  discreteStep?: number;
};

export type GapAwarePlotData = Data & {
  mltraceGapPolicy?: PlotGapPolicy;
};

export function withLineGapPolicy(data: Data, policy: PlotGapPolicy): GapAwarePlotData {
  return { ...data, mltraceGapPolicy: policy } as GapAwarePlotData;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle];
}

function timestampMillis(value: unknown): number | null {
  if (value instanceof Date) {
    const parsed = value.getTime();
    return Number.isFinite(parsed) ? parsed : null;
  }
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}(?:[T\s]|$)/.test(value)) return null;
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : null;
}

function finitePlotValue(value: unknown): unknown {
  return typeof value === 'number' && !Number.isFinite(value) ? null : value;
}

function expandAlignedArray(value: unknown, originalLength: number, breakBefore: Set<number>): unknown {
  if (!Array.isArray(value) || value.length !== originalLength) return value;
  const expanded: unknown[] = [];
  value.forEach((item, index) => {
    if (breakBefore.has(index)) expanded.push(null);
    expanded.push(finitePlotValue(item));
  });
  return expanded;
}

function expandNestedArrays(
  value: unknown,
  originalLength: number,
  breakBefore: Set<number>,
  keys: string[],
): unknown {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return value;
  const record = value as Record<string, unknown>;
  let changed = false;
  const next = { ...record };
  keys.forEach((key) => {
    const expanded = expandAlignedArray(record[key], originalLength, breakBefore);
    if (expanded !== record[key]) {
      next[key] = expanded;
      changed = true;
    }
  });
  return changed ? next : value;
}

function midpoint(left: unknown, right: unknown): unknown {
  const leftTime = timestampMillis(left);
  const rightTime = timestampMillis(right);
  if (leftTime !== null && rightTime !== null) return new Date((leftTime + rightTime) / 2).toISOString();
  if (typeof left === 'number' && typeof right === 'number') return (left + right) / 2;
  return null;
}

function inferredTimeBreaks(x: unknown[]): Set<number> {
  const parsed = x.map(timestampMillis);
  if (parsed.some((value) => value === null)) return new Set();
  const times = parsed as number[];
  const deltas = times.slice(1).map((value, index) => value - times[index]).filter((value) => value > 0);
  const typical = median(deltas);
  if (typical === null) return new Set();
  const gapThreshold = Math.max(15_000, typical * 5);
  return new Set(times.flatMap((value, index) => (
    index > 0 && value - times[index - 1] > gapThreshold ? [index] : []
  )));
}

function lineBreaks(x: unknown[], policy: PlotGapPolicy | undefined): Set<number> {
  if (policy?.continuity && policy.continuity.length === x.length) {
    return new Set(policy.continuity.flatMap((segment, index) => (
      index > 0 && segment !== policy.continuity?.[index - 1] ? [index] : []
    )));
  }
  if (policy?.discreteStep && policy.discreteStep > 0 && x.every((value) => typeof value === 'number' && Number.isFinite(value))) {
    return new Set((x as number[]).flatMap((value, index) => (
      index > 0 && value - (x[index - 1] as number) > policy.discreteStep! * 5 ? [index] : []
    )));
  }
  return inferredTimeBreaks(x);
}

export function preparePlotData(data: Data[]): Data[] {
  return data.map((input) => {
    const trace = input as GapAwarePlotData & Record<string, unknown>;
    const { mltraceGapPolicy: policy, ...plotlyTrace } = trace;
    const mode = typeof trace.mode === 'string' ? trace.mode : '';
    if (trace.type !== 'scatter' || !mode.includes('lines')) return plotlyTrace as Data;

    const x = Array.isArray(trace.x) ? [...trace.x] : null;
    const y = Array.isArray(trace.y) ? trace.y.map(finitePlotValue) : null;
    if (!x || !y || x.length !== y.length) {
      return { ...plotlyTrace, connectgaps: false, ...(y ? { y } : {}) } as Data;
    }

    const breakBefore = lineBreaks(x, policy);
    const expandedX: unknown[] = [];
    const expandedY: unknown[] = [];
    x.forEach((value, index) => {
      if (breakBefore.has(index)) {
        expandedX.push(midpoint(x[index - 1], value));
        expandedY.push(null);
      }
      expandedX.push(value);
      expandedY.push(y[index]);
    });

    const prepared: Record<string, unknown> = {
      ...plotlyTrace,
      x: expandedX,
      y: expandedY,
      connectgaps: false,
    };
    ['text', 'hovertext', 'ids', 'customdata'].forEach((key) => {
      if (plotlyTrace[key] !== undefined) {
        prepared[key] = expandAlignedArray(plotlyTrace[key], x.length, breakBefore);
      }
    });
    if (plotlyTrace.marker !== undefined) {
      prepared.marker = expandNestedArrays(plotlyTrace.marker, x.length, breakBefore, ['size', 'color', 'symbol', 'opacity']);
    }
    if (plotlyTrace.error_x !== undefined) {
      prepared.error_x = expandNestedArrays(plotlyTrace.error_x, x.length, breakBefore, ['array', 'arrayminus']);
    }
    if (plotlyTrace.error_y !== undefined) {
      prepared.error_y = expandNestedArrays(plotlyTrace.error_y, x.length, breakBefore, ['array', 'arrayminus']);
    }
    return prepared as Data;
  });
}
