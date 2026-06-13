import { Alert, Badge, Paper, ScrollArea, Stack, Text, Title } from '@mantine/core';
import { ArrowRight, Shuffle } from 'lucide-react';

import type {
  MethodConfiguration,
  PreprocessingGraphNode,
  PreprocessingPipeline,
  TrainingDataset,
} from '../types';

/** Order the linear preprocessing chain by following its edges (load_image first). */
function orderedGraphNodes(pipeline: PreprocessingPipeline): PreprocessingGraphNode[] {
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

function stepDetail(node: PreprocessingGraphNode): string {
  const interesting = ['width', 'height', 'mode', 'method', 'dtype', 'x', 'y'];
  const parts = interesting
    .filter((key) => node.config[key] !== undefined)
    .map((key) => `${key}=${node.config[key]}`);
  return parts.length > 0 ? parts.join(', ') : 'default config';
}

function FlowNode({ section, label, detail }: { section: string; label: string; detail: string }) {
  return (
    <Paper withBorder p="sm" radius="sm" className={`model-diagram-node model-section-${section}`}>
      <Stack gap={4}>
        <Badge size="xs" variant="light">
          {section}
        </Badge>
        <Text fw={700} size="sm">
          {label}
        </Text>
        <Text size="xs" c="dimmed">
          {detail}
        </Text>
      </Stack>
    </Paper>
  );
}

function FlowArrow() {
  return <ArrowRight size={18} className="model-diagram-arrow" />;
}

/**
 * Horizontal end-to-end visualization of the composed training pipeline:
 * training sets -> preprocessing steps -> method architecture (stored diagram).
 */
export function TrainingPipelineFlow({
  trainingDatasets,
  shuffle,
  preprocessingPipeline,
  configuration,
}: {
  trainingDatasets: TrainingDataset[];
  shuffle: boolean;
  preprocessingPipeline: PreprocessingPipeline | null;
  configuration: MethodConfiguration | null;
}) {
  const empty =
    trainingDatasets.length === 0 && preprocessingPipeline === null && configuration === null;

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Title order={3}>Pipeline Overview</Title>
        {empty && (
          <Alert color="blue">
            Select training sets, a preprocessing pipeline, and a method to see the full pipeline here.
          </Alert>
        )}
        {!empty && (
          <ScrollArea>
            <div className="model-diagram-row">
              {trainingDatasets.map((dataset, index) => (
                <div key={`dataset-${dataset.id}`} className="model-diagram-item">
                  <FlowNode
                    section="dataset"
                    label={dataset.name}
                    detail={`${dataset.total_selected_images} images · ${dataset.dataset_names.join(', ')}`}
                  />
                  {index < trainingDatasets.length - 1 && <FlowArrow />}
                </div>
              ))}
              {trainingDatasets.length > 0 && (
                <div className="model-diagram-item">
                  <Badge
                    variant={shuffle ? 'filled' : 'outline'}
                    color="teal"
                    leftSection={<Shuffle size={12} />}
                  >
                    {shuffle ? 'shuffled' : 'in order'}
                  </Badge>
                  {(preprocessingPipeline || configuration) && <FlowArrow />}
                </div>
              )}
              {preprocessingPipeline &&
                orderedGraphNodes(preprocessingPipeline).map((node, index, all) => (
                  <div key={`step-${node.id}`} className="model-diagram-item">
                    <FlowNode section="preprocess" label={node.type} detail={stepDetail(node)} />
                    {(index < all.length - 1 || configuration) && <FlowArrow />}
                  </div>
                ))}
              {configuration &&
                (configuration.diagram?.nodes ?? []).map((node, index, all) => (
                  <div key={`method-${node.id}`} className="model-diagram-item">
                    <FlowNode section={node.section} label={node.label} detail={node.detail} />
                    {index < all.length - 1 && <FlowArrow />}
                  </div>
                ))}
            </div>
          </ScrollArea>
        )}
      </Stack>
    </Paper>
  );
}
