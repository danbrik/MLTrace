import type {
  MethodConfiguration,
  PreprocessingGraphNode,
  PreprocessingPipeline,
  TrainingDataset,
} from '../types';

/** Order a linear preprocessing chain by following its edges (load_image first). */
export function orderedGraphNodes(pipeline: PreprocessingPipeline): PreprocessingGraphNode[] {
  const { nodes, edges } = pipeline.graph;
  if (edges.length === 0) return nodes;
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const nextBySource = new Map(edges.map((edge) => [edge.source, edge.target]));
  const targets = new Set(edges.map((edge) => edge.target));
  let currentId = nodes.find((node) => !targets.has(node.id))?.id;
  const ordered: PreprocessingGraphNode[] = [];
  while (currentId && nodeById.has(currentId) && ordered.length < nodes.length) {
    ordered.push(nodeById.get(currentId)!);
    currentId = nextBySource.get(currentId);
  }
  return ordered.length === nodes.length ? ordered : nodes;
}

/** Compact one-line summary of a preprocessing step's most relevant config values. */
export function stepDetail(node: PreprocessingGraphNode): string {
  const interesting = ['width', 'height', 'mode', 'method', 'dtype', 'x', 'y'];
  const parts = interesting
    .filter((key) => node.config[key] !== undefined)
    .map((key) => `${key}=${node.config[key]}`);
  return parts.length > 0 ? parts.join(', ') : 'default config';
}

// --- Image-size helpers -----------------------------------------------------
// The size chain that drives cross-filtering across the three pickers:
//   dataset image size == preprocessing input size
//   preprocessing output size == method input size

/** Format a width/height pair as "WxH", or null when either is missing. */
export function formatResolution(width: number | null | undefined, height: number | null | undefined): string | null {
  if (typeof width === 'number' && typeof height === 'number') return `${width}x${height}`;
  return null;
}

export function pipelineInputResolution(pipeline: PreprocessingPipeline): string | null {
  return formatResolution(pipeline.input_width, pipeline.input_height);
}

export function pipelineOutputResolution(pipeline: PreprocessingPipeline): string | null {
  return formatResolution(pipeline.output_width, pipeline.output_height);
}

export function methodInputResolution(configuration: MethodConfiguration): string | null {
  const config = configuration.method_config ?? {};
  return formatResolution(config.input_width as number, config.input_height as number);
}

export function datasetResolutions(dataset: TrainingDataset): string[] {
  return dataset.image_resolutions ?? [];
}

/**
 * The shared image-size signature of the selected datasets: sorted unique
 * resolutions across the datasets that have a known size. Returns null when no
 * selected dataset has a recorded resolution (i.e. no active size constraint).
 */
export function datasetSizeSignature(
  selectedIds: number[],
  datasetById: Map<number, TrainingDataset>,
): string[] | null {
  const resolutions = new Set<string>();
  for (const id of selectedIds) {
    const dataset = datasetById.get(id);
    if (!dataset) continue;
    datasetResolutions(dataset).forEach((resolution) => resolutions.add(resolution));
  }
  return resolutions.size > 0 ? [...resolutions].sort() : null;
}
