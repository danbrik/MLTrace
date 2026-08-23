import { useCallback, useEffect, useRef, useState } from 'react';
import type { PlotRelayoutEvent } from './plotly';

export type TimeSeriesAxisRange = [number, number];
export type TimeSeriesTraceValues = {
  x: Array<string | number | Date | null>;
  y: Array<number | null>;
  yaxis: string;
};

export function medianPositiveTimeDelta(timestamps: string[]): number | null {
  const ordered = timestamps
    .map((timestamp, index) => ({ timestamp, index, time: new Date(timestamp).getTime() }))
    .filter((item) => Number.isFinite(item.time))
    .sort((left, right) => left.time - right.time || left.index - right.index);
  const deltas: number[] = [];
  for (let index = 1; index < ordered.length; index += 1) {
    const delta = ordered[index].time - ordered[index - 1].time;
    if (delta > 0) deltas.push(delta);
  }
  if (deltas.length === 0) return null;
  deltas.sort((left, right) => left - right);
  const middle = Math.floor(deltas.length / 2);
  return deltas.length % 2 === 0 ? (deltas[middle - 1] + deltas[middle]) / 2 : deltas[middle];
}

export function paddedAxisRange(minimum: number, maximum: number): TimeSeriesAxisRange {
  if (minimum === maximum) {
    const padding = Math.max(Math.abs(minimum) * 0.05, 1e-9);
    return [minimum - padding, maximum + padding];
  }
  const padding = (maximum - minimum) * 0.05;
  return [minimum - padding, maximum + padding];
}

export function visibleAxisRanges(
  traces: TimeSeriesTraceValues[],
  xRange: TimeSeriesAxisRange | null,
): Record<string, TimeSeriesAxisRange> {
  const valuesByAxis = new Map<string, number[]>();
  for (const trace of traces) {
    trace.y.forEach((value, index) => {
      if (value === null || !Number.isFinite(value)) return;
      const timestamp = new Date(trace.x[index] ?? '').getTime();
      if (!Number.isFinite(timestamp)) return;
      if (xRange && (timestamp < xRange[0] || timestamp > xRange[1])) return;
      const values = valuesByAxis.get(trace.yaxis) ?? [];
      values.push(value);
      valuesByAxis.set(trace.yaxis, values);
    });
  }
  return Object.fromEntries([...valuesByAxis].flatMap(([axis, values]) => {
    if (values.length === 0) return [];
    let minimum = values[0];
    let maximum = values[0];
    for (const value of values.slice(1)) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    return [[axis, paddedAxisRange(minimum, maximum)]];
  }));
}

function rangesEqual(left: TimeSeriesAxisRange | undefined, right: TimeSeriesAxisRange): boolean {
  return left !== undefined && left[0] === right[0] && left[1] === right[1];
}

export function useVisibleAutomaticYRanges(traces: TimeSeriesTraceValues[]) {
  const tracesRef = useRef(traces);
  const visibleXRangeRef = useRef<TimeSeriesAxisRange | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [automaticYRanges, setAutomaticYRanges] = useState<Record<string, TimeSeriesAxisRange>>(
    () => visibleAxisRanges(traces, null),
  );
  tracesRef.current = traces;

  const scheduleAutomaticYRanges = useCallback((xRange: TimeSeriesAxisRange | null) => {
    // Store synchronously so a burst of wheel, pan, or range-slider events
    // always resolves against the newest viewport in the queued frame.
    visibleXRangeRef.current = xRange;
    if (animationFrameRef.current !== null) return;
    animationFrameRef.current = window.requestAnimationFrame(() => {
      animationFrameRef.current = null;
      const nextRanges = visibleAxisRanges(tracesRef.current, visibleXRangeRef.current);
      // A viewport that only contains a gap keeps the last meaningful scale.
      if (Object.keys(nextRanges).length === 0) return;
      setAutomaticYRanges((current) => {
        const changed = Object.entries(nextRanges).some(([axis, range]) => !rangesEqual(current[axis], range));
        return changed ? { ...current, ...nextRanges } : current;
      });
    });
  }, []);

  useEffect(() => {
    scheduleAutomaticYRanges(visibleXRangeRef.current);
  }, [scheduleAutomaticYRanges, traces]);

  useEffect(() => () => {
    if (animationFrameRef.current !== null) window.cancelAnimationFrame(animationFrameRef.current);
  }, []);

  return { automaticYRanges, scheduleAutomaticYRanges };
}

export function relayoutRange(
  event: PlotRelayoutEvent,
  axis: 'xaxis' | string,
): TimeSeriesAxisRange | null | undefined {
  const values = event as Record<string, unknown>;
  const direct = values[`${axis}.range`];
  if (Array.isArray(direct) && direct.length >= 2) {
    const start = axis === 'xaxis' ? new Date(String(direct[0])).getTime() : Number(direct[0]);
    const end = axis === 'xaxis' ? new Date(String(direct[1])).getTime() : Number(direct[1]);
    return Number.isFinite(start) && Number.isFinite(end) ? [Math.min(start, end), Math.max(start, end)] : undefined;
  }
  const rawStart = values[`${axis}.range[0]`];
  const rawEnd = values[`${axis}.range[1]`];
  if (rawStart !== undefined && rawEnd !== undefined) {
    const start = axis === 'xaxis' ? new Date(String(rawStart)).getTime() : Number(rawStart);
    const end = axis === 'xaxis' ? new Date(String(rawEnd)).getTime() : Number(rawEnd);
    return Number.isFinite(start) && Number.isFinite(end) ? [Math.min(start, end), Math.max(start, end)] : undefined;
  }
  if (values[`${axis}.autorange`] === true) return null;
  return undefined;
}
