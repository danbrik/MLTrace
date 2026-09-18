import type { Data } from '../lib/plotly';
import type { CharacterizationHistogram, CharacterizationSeries, CharacterizationSummary } from '../types';

export const summaryColumns: { key: keyof CharacterizationSummary; label: string }[] = [
  { key: 'variable', label: 'Variable' }, { key: 'data_type', label: 'Type' }, { key: 'valid_n', label: 'N valid' },
  ...['mean', 'median', 'std', 'iqr', 'min', 'q01', 'q05', 'q25', 'q75', 'q95', 'q99', 'max'].map(key => ({
    key: key as keyof CharacterizationSummary, label: key.startsWith('q') || ['std', 'iqr'].includes(key) ? key.toUpperCase() : key[0].toUpperCase() + key.slice(1),
  })),
];
export function summaryRows(rows: CharacterizationSummary[], search: string, type: string, sort: keyof CharacterizationSummary, descending: boolean) {
  return rows.filter(row => row.variable.toLowerCase().includes(search.toLowerCase()) && (type === 'all' || row.data_type === type)).sort((a, b) => {
    const left = a[sort], right = b[sort];
    if (left == null) return right == null ? 0 : 1;
    if (right == null) return -1;
    const value = typeof left === 'number' && typeof right === 'number' ? left - right : String(left).toLowerCase() < String(right).toLowerCase() ? -1 : String(left).toLowerCase() > String(right).toLowerCase() ? 1 : 0;
    return descending ? -value : value;
  });
}
export function histogramTrace(hist: CharacterizationHistogram, name: string): Data[] {
  return [{ type: 'bar', name, x: hist.counts.map((_, i) => (hist.edges[i] + hist.edges[i + 1]) / 2),
    y: hist.counts, width: hist.counts.map((_, i) => hist.edges[i + 1] - hist.edges[i]),
    customdata: hist.counts.map((_, i) => [hist.edges[i], hist.edges[i + 1]]), marker: { color: '#579aeb' },
    hovertemplate: '[%{customdata[0]}, %{customdata[1]}]<br>Count: %{y}<extra></extra>' } as Data];
}
export function temporalTraces(series: CharacterizationSeries): Data[] {
  const x: (string | null)[] = [], y: (number | null)[] = [];
  const custom: (string | number)[][] = [];
  for (const point of series.points) {
    if (x.length && !point.connect_previous) { x.push(null); y.push(null); custom.push([]); }
    x.push(point.time); y.push(point.value);
    custom.push([point.end, point.valid_n, point.q25 ?? '', point.q75 ?? '']);
  }
  const data: Data[] = [];
  if (series.aggregated) {
    const bandX: (string | null)[] = [], bandY: (number | null)[] = [];
    let segment: CharacterizationSeries['points'] = [];
    const flush = () => {
      if (segment.length > 1) {
        bandX.push(...segment.map(p => p.time), ...[...segment].reverse().map(p => p.time), segment[0].time, null);
        bandY.push(...segment.map(p => p.q25), ...[...segment].reverse().map(p => p.q75), segment[0].q25, null);
      }
      segment = [];
    };
    for (const point of series.points) {
      if (!point.connect_previous || point.value == null) flush();
      if (point.value != null) segment.push(point);
    }
    flush();
    // Independent closed polygons: tonexty fills can bridge gaps even when lines do not.
    data.push({ type: 'scatter', mode: 'none', x: bandX, y: bandY, fill: 'toself',
      fillcolor: 'rgba(34,139,230,0.18)', connectgaps: false, name: 'Q25–Q75', hoverinfo: 'skip' } as Data);
  }
  data.push({ type: 'scatter', mode: 'lines+markers', name: series.aggregated ? 'Median' : 'Original value', x, y,
    customdata: custom, connectgaps: false, marker: { size: 3 }, line: { color: '#228be6', width: 1.5 },
    hovertemplate: '%{x} – %{customdata[0]}<br>Value: %{y}<br>N valid: %{customdata[1]}<br>Q25: %{customdata[2]} · Q75: %{customdata[3]}<extra></extra>' } as Data);
  // Isolated aggregated bins still expose their spread without connecting across gaps.
  if (series.aggregated) data.push({ type: 'scatter', mode: 'markers', showlegend: false, hoverinfo: 'skip',
    x: series.points.map(p => p.time), y: series.points.map(p => p.value), marker: { size: 2, color: '#228be6' },
    error_y: { type: 'data', symmetric: false, array: series.points.map(p => p.q75 == null || p.value == null ? 0 : p.q75 - p.value),
      arrayminus: series.points.map(p => p.q25 == null || p.value == null ? 0 : p.value - p.q25), thickness: 1, width: 0 } } as Data);
  return data;
}
