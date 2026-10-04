import type { TrainingRun, TrainingRunMetric } from '../types';
const builders = new Set(['sequential_autoencoder', 'sequential_spatial_autoencoder', 'sequential_variational_autoencoder', 'spatiotemporal_autoencoder']);
export function supportsLossPlot(run: Pick<TrainingRun, 'training_mode' | 'builder_kind'>): boolean {
  return run.training_mode === 'gradient' && builders.has(run.builder_kind);
}
export function lossPlotRevision(run: TrainingRun): string {
  return JSON.stringify([run.enqueued_at, run.started_at, run.metrics]);
}
export function lossMetricsCsv(metrics: TrainingRunMetric[]): string {
  const value = (n: number | null) => n != null && Number.isFinite(n) ? String(n) : '';
  return 'epoch,train_loss,val_loss\r\n' + [...metrics].sort((a, b) => a.epoch - b.epoch)
    .map(m => `${m.epoch},${value(m.train_loss)},${value(m.val_loss)}\r\n`).join('');
}
export function downloadLossCsv(runId: number, metrics: TrainingRunMetric[]) {
  const url = URL.createObjectURL(new Blob([lossMetricsCsv(metrics)], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url; link.download = `training-run-${runId}-loss.csv`;
  document.body.appendChild(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
