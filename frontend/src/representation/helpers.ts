import type { RepresentationConfig, RepresentationInterval, RepresentationLabel, RepresentationPoint } from '../types';
import type { Data } from '../lib/plotly';

export const LABELS: Record<RepresentationLabel, string> = { normal: 'Normal', anomaly: 'Anomalie', buffer: 'Puffer' };
export const PHASES: Record<string, string> = {
  queued: 'Wartet im Scheduler', selection: 'Bildauswahl wird geladen', loading_model: 'DINOv3 wird geladen',
  features: 'Preprocessing und Featureextraktion', pca: 'PCA wird berechnet', umap: 'UMAP wird berechnet',
  kmeans: 'KMeans wird berechnet', evaluation: 'Vergleich mit Ground Truth', saving: 'Ergebnisse werden gespeichert',
  finished: 'Analyse abgeschlossen', failed: 'Analyse fehlgeschlagen', aborted: 'Analyse abgebrochen',
};

export function nextEventId(intervals: RepresentationInterval[]): string {
  return `A${Math.max(0, ...intervals.map(item => Number(item.event_id?.slice(1) ?? 0))) + 1}`;
}

export function validateConfig(config: RepresentationConfig): string | null {
  if (!config.training_dataset_id || !config.preprocessing_pipeline_id) return 'Datensatz und Preprocessing-Pipeline auswählen.';
  if (!config.intervals.length) return 'Mindestens einen Bereich hinzufügen.';
  if (!Number.isInteger(config.cluster_count) || config.cluster_count < 2) return 'Clusterzahl muss eine ganze Zahl ab 2 sein.';
  if (!(config.pca_variance > 0 && config.pca_variance < 1)) return 'PCA-Varianz muss zwischen 0 und 100 % liegen.';
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 4294967295) return 'Seed muss eine ganze Zahl zwischen 0 und 4294967295 sein.';
  const ordered = [...config.intervals].sort((a, b) => a.start.localeCompare(b.start));
  for (const [index, item] of ordered.entries()) {
    if (!item.name.trim() || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(item.start) ||
        !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(item.end) || item.end <= item.start) return 'Für jeden Bereich Namen und gültigen Beginn/Ende angeben.';
    if (!Number.isInteger(item.sampling_rate) || item.sampling_rate < 1) return 'Samplingrate muss eine ganze Zahl ab 1 sein.';
    if (index > 0 && item.start < ordered[index - 1].end) return 'Bereiche dürfen sich nicht überlappen.';
  }
  return null;
}

const COLORS: Record<string, string> = { Normal: '#228be6', Anomalie: '#e03131', Puffer: '#868e96' };
const EVENTS = ['#e03131', '#ae3ec9', '#f08c00', '#099268', '#6741d9', '#c2255c', '#5c940d', '#0b7285'];
export function projectionTraces(points: RepresentationPoint[], projection: 'pca' | 'umap', byEvent: boolean): Data[] {
  const groups = new Map<string, RepresentationPoint[]>();
  for (const point of points) {
    const key = byEvent && point.label === 'anomaly' ? point.event_id ?? 'Anomalie' : LABELS[point.label];
    const rows = groups.get(key) ?? [];
    rows.push(point);
    groups.set(key, rows);
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true })).map(([name, rows]) => ({
    type: 'scatter', mode: 'markers', name,
    x: rows.map(row => projection === 'pca' ? row.pca_x : row.umap_x),
    y: rows.map(row => projection === 'pca' ? row.pca_y : row.umap_y),
    customdata: rows.map(row => [row.timestamp, LABELS[row.label], row.interval_name, row.event_id ?? '—']),
    hovertemplate: '%{customdata[0]}<br>%{customdata[1]}<br>Bereich: %{customdata[2]}<br>Ereignis: %{customdata[3]}<extra></extra>',
    marker: { size: 7, opacity: byEvent && (name === 'Normal' || name === 'Puffer') ? 0.35 : 0.8,
      color: COLORS[name] ?? EVENTS[(Number(name.slice(1)) - 1) % EVENTS.length] },
  } as Data));
}
