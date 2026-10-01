import { getFullTestingRunPlotSeries } from '../api';
import type { TestingRun } from '../types';
import { buildInferenceExport, type ExportOptions, type InferenceExportSource } from './export';

export async function loadInferenceExport(
  runs: Array<Pick<TestingRun, 'id' | 'name' | 'status'>>,
  options: ExportOptions,
  projectId: string,
  signal?: AbortSignal,
  onProgress?: (completed: number) => void,
) {
  const sources: InferenceExportSource[] = [];
  // Bound memory/network pressure from many simultaneous full-resolution requests.
  for (const run of runs) {
    signal?.throwIfAborted();
    if (run.status !== 'finished') throw new Error(`Inference "${run.name}" is no longer finished. Refresh the source list.`);
    try {
      const result = await getFullTestingRunPlotSeries(run.id, { score_series: 'score', projectId, signal });
      if (result.points.length !== result.total) throw new Error('The stored results changed or could not be loaded completely. Create the preview again.');
      if (!result.points.length) throw new Error('No stored inference points.');
      sources.push({ id: run.id, name: run.name, points: result.points });
      onProgress?.(sources.length);
    } catch (error) {
      throw new Error(`Inference "${run.name}" (#${run.id}): ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  signal?.throwIfAborted();
  return buildInferenceExport(sources, options);
}
