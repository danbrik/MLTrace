import type { Data, Layout } from '../lib/plotly';
import { rangeProblem } from '../timeRangePresets/helpers';
import type { Config, PlotSettings, Role, Summary } from './types';
export const roles: Role[] = ['reference', 'comparison'];
export const labels: Record<Role, string> = { reference: 'Referenz', comparison: 'Vergleich' };
export const defaultConfig: Config = {
  training_dataset_id: 0, preprocessing_pipeline_id: 0,
  reference: { start: '', end: '' }, comparison: { start: '', end: '' }, deltas_seconds: [1, 2, 5, 15, 30, 60],
};
export const defaultPlot: PlotSettings = {
  title: 'Zeitabstands-Analyse', x_title: 'Zeitabstand Δt (s)', y_title: 'Mittlere absolute Pixeländerung (Pipeline-Einheiten)',
  x_range: null, y_range: null, reference_color: '#1971c2', comparison_color: '#e8590c',
};
export const phaseLabel = (value: string) => ({ queued: 'Wartend', running: 'Berechnung läuft', finished: 'Fertig', aborted: 'Abgebrochen',
  failed: 'Fehlgeschlagen', validating: 'Auswahl prüfen', calculating: 'Bildpaare berechnen', summarizing: 'Statistik berechnen', exporting: 'Tabellen speichern' }[value] ?? value);
export const displayTime = (value: string) => value.replace('T', ' ');
export const numberText = (value: number | null) => value === null ? '—' : value.toLocaleString('de-DE', { maximumSignificantDigits: 8 });
export function validDelta(value: number) { return Number.isSafeInteger(value) && value > 0; }
export function addDelta(current: number[], value: number) {
  return validDelta(value) ? [...new Set([...current, value])].sort((a, b) => a - b) : current;
}
export function validateConfig(config: Config, min?: string, max?: string) {
  if (!config.training_dataset_id || !config.preprocessing_pipeline_id) return 'Bitte Datensatz und Preprocessing auswählen.';
  for (const role of roles) {
    const problem = rangeProblem(config[role], min, max);
    if (problem) return `${labels[role]}: ${problem}`;
  }
  if (!config.deltas_seconds.length || !config.deltas_seconds.every(validDelta)) return 'Mindestens einen positiven ganzzahligen Zeitabstand eingeben.';
  if (new Set(config.deltas_seconds).size !== config.deltas_seconds.length) return 'Zeitabstände dürfen nicht doppelt vorkommen.';
  return null;
}
export function validatePlot(settings: PlotSettings) {
  if ([settings.title, settings.x_title, settings.y_title].some(value => value.length > 250)) return 'Titel dürfen höchstens 250 Zeichen enthalten.';
  if (![settings.reference_color, settings.comparison_color].every(value => /^#[0-9a-fA-F]{6}$/.test(value))) return 'Bitte gültige Farben auswählen.';
  for (const range of [settings.x_range, settings.y_range]) {
    if (range && (!Number.isFinite(range.minimum) || !Number.isFinite(range.maximum) || range.minimum >= range.maximum)) return 'Die Achsenuntergrenze muss kleiner als die Obergrenze sein.';
  }
  return null;
}
const rgba = (hex: string) => `rgba(${[1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16)).join(',')},0.16)`;
const plainTitle = (value: string) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
export function plotData(summary: Summary[], settings: PlotSettings): Data[] {
  return roles.flatMap(role => {
    const rows = summary.filter(row => row.role === role).sort((a, b) => a.delta_seconds - b.delta_seconds);
    const color = settings[`${role}_color`];
    const common = { type: 'scatter' as const, legendgroup: role, connectgaps: false };
    // Separate bands at missing values: Plotly fill can otherwise bridge nulls.
    const segments: Summary[][] = [];
    let segment: Summary[] = [];
    for (const row of rows) {
      if (row.q1 === null || row.q3 === null) { if (segment.length) segments.push(segment); segment = []; }
      else segment.push(row);
    }
    if (segment.length) segments.push(segment);
    const bands = segments.flatMap(group => [
      { ...common, x: group.map(row => row.delta_seconds), y: group.map(row => row.q1), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { ...common, x: group.map(row => row.delta_seconds), y: group.map(row => row.q3), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: rgba(color), showlegend: false, hoverinfo: 'skip' },
    ] as Data[]);
    return [...bands, { ...common, x: rows.map(row => row.delta_seconds), y: rows.map(row => row.median), mode: 'lines+markers', name: labels[role], line: { color }, marker: { color },
      customdata: rows.map(row => [row.pair_count, row.q1, row.q3]),
      hovertemplate: '%{x} s<br>Median: %{y}<br>Paare: %{customdata[0]}<br>Q1: %{customdata[1]}<br>Q3: %{customdata[2]}<extra>%{fullData.name}</extra>' } as Data];
  });
}
export function plotLayout(settings: PlotSettings): Partial<Layout> {
  return { title: { text: plainTitle(settings.title) }, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { color: '#212529', family: 'Arial, sans-serif', size: 14 },
    margin: { l: 95, r: 30, t: 65, b: 95 }, showlegend: true, legend: { orientation: 'h', y: -0.2 },
    xaxis: { type: 'linear', title: { text: plainTitle(settings.x_title) }, gridcolor: '#e9ecef',
      autorange: !settings.x_range, range: settings.x_range ? [settings.x_range.minimum, settings.x_range.maximum] : undefined },
    yaxis: { type: 'linear', title: { text: plainTitle(settings.y_title) }, gridcolor: '#e9ecef',
      autorange: !settings.y_range, range: settings.y_range ? [settings.y_range.minimum, settings.y_range.maximum] : undefined },
  };
}
