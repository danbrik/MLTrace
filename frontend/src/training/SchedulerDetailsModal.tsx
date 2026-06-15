import { Badge, Group, Modal, Paper, ScrollArea, Stack, Table, Text } from '@mantine/core';
import type { ReactNode } from 'react';

import { methodLabel } from '../methods/utils';
import { orderedGraphNodes, stepDetail, formatResolution } from './graph';
import { formatDuration } from './runStatus';
import type {
  MethodConfiguration,
  MethodDefinition,
  PreprocessingPipeline,
  TestingRun,
  TrainingPipeline,
  TrainingRun,
} from '../types';

export type SchedulerJob =
  | { kind: 'train'; run: TrainingRun }
  | { kind: 'test'; run: TestingRun };

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Group gap="sm" align="flex-start" wrap="nowrap">
      <Text size="xs" fw={700} tt="uppercase" c="dimmed" w={130} style={{ flexShrink: 0 }}>
        {label}
      </Text>
      <div style={{ flex: 1 }}>{children}</div>
    </Group>
  );
}

function PreprocessingSteps({ pipeline }: { pipeline: PreprocessingPipeline | undefined }) {
  if (!pipeline) return <Text size="sm" c="dimmed">unavailable</Text>;
  return (
    <Stack gap={4}>
      <Text size="sm">
        {pipeline.name}{' '}
        <Text span size="xs" c="dimmed">
          · in {formatResolution(pipeline.input_width, pipeline.input_height) ?? 'n/a'} → out{' '}
          {formatResolution(pipeline.output_width, pipeline.output_height) ?? 'n/a'}
        </Text>
      </Text>
      <Group gap={6}>
        {orderedGraphNodes(pipeline).map((node, index) => (
          <Badge key={node.id} size="sm" variant="light" color="gray">
            {index + 1}. {node.type}{' '}
            <Text span size="xs" c="dimmed">
              ({stepDetail(node)})
            </Text>
          </Badge>
        ))}
      </Group>
    </Stack>
  );
}

function MethodDetail({
  configuration,
  definition,
}: {
  configuration: MethodConfiguration | undefined;
  definition: MethodDefinition | undefined;
}) {
  if (!configuration) return <Text size="sm" c="dimmed">unavailable</Text>;
  const config = configuration.method_config ?? {};
  const keys = ['input_channels', 'input_width', 'input_height', 'latent_dim', 'kl_weight', 'output_activation', 'aggregation', 'accumulator_dtype'];
  return (
    <Stack gap={4}>
      <Group gap="xs">
        <Text size="sm">{methodLabel(definition, configuration.method_type)}</Text>
      </Group>
      <Group gap={6}>
        {keys
          .filter((key) => config[key] !== undefined)
          .map((key) => (
            <Badge key={key} size="sm" variant="light" color="indigo">
              {key}={String(config[key])}
            </Badge>
          ))}
      </Group>
      {(configuration.diagram?.nodes?.length ?? 0) > 0 && (
        <Group gap={6}>
          {configuration.diagram.nodes.map((node, index) => (
            <Badge key={node.id} size="sm" variant="light" className={`model-section-${node.section}`}>
              {index + 1}. {node.label}
            </Badge>
          ))}
        </Group>
      )}
    </Stack>
  );
}

export function SchedulerDetailsModal({
  job,
  onClose,
  pipelineById,
  preprocessingById,
  methodById,
  methodByType,
  trainingRunById,
}: {
  job: SchedulerJob | null;
  onClose: () => void;
  pipelineById: Map<number, TrainingPipeline>;
  preprocessingById: Map<number, PreprocessingPipeline>;
  methodById: Map<number, MethodConfiguration>;
  methodByType: Map<string, MethodDefinition>;
  trainingRunById: Map<number, TrainingRun>;
}) {
  const title = job ? (job.kind === 'train' ? job.run.training_pipeline_name : job.run.name) : '';

  function renderTraining(run: TrainingRun) {
    const pipeline = pipelineById.get(run.training_pipeline_id);
    const preprocessing = pipeline ? preprocessingById.get(pipeline.preprocessing_pipeline_id) : undefined;
    const configuration = pipeline ? methodById.get(pipeline.method_configuration_id) : undefined;
    return (
      <Stack gap="sm">
        <Row label="Type">
          <Badge color="blue" variant="light">Training</Badge>
        </Row>
        <Row label="Datasets">
          <Stack gap={2}>
            {(pipeline?.training_datasets ?? []).map((entry) => (
              <Text key={entry.training_dataset_id} size="sm">
                {entry.name}{' '}
                <Text span size="xs" c="dimmed">· {entry.total_selected_images} images · {entry.dataset_names.join(', ')}</Text>
              </Text>
            ))}
            <Badge size="sm" variant={run.shuffle ? 'filled' : 'outline'} color="teal" w="fit-content">
              {run.shuffle ? 'shuffled' : 'in order'}
            </Badge>
          </Stack>
        </Row>
        <Row label="Preprocessing"><PreprocessingSteps pipeline={preprocessing} /></Row>
        <Row label="Method"><MethodDetail configuration={configuration} definition={methodByType.get(run.method_type)} /></Row>
        <Row label="Parameters">
          <Group gap={6}>
            {Object.entries(run.training_parameters ?? {}).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">{key}={String(value)}</Badge>
            ))}
            {Object.keys(run.training_parameters ?? {}).length === 0 && <Text size="sm" c="dimmed">none</Text>}
          </Group>
        </Row>
        <Row label="Result">
          <Text size="sm">
            {run.epochs_completed}/{run.epochs_total ?? '?'} epochs · train {run.train_loss?.toFixed(4) ?? '—'} · val{' '}
            {run.val_loss?.toFixed(4) ?? '—'} · {formatDuration(run.duration_seconds)} · {run.device ?? '—'}
          </Text>
        </Row>
        <Row label="Artifact">
          <Text size="xs" className="path-text">{run.artifact_kind ?? '—'}: {run.artifact_path ?? '—'}</Text>
        </Row>
        {run.error_message && (
          <Paper withBorder p="xs" radius="sm" bg="var(--mantine-color-red-0)">
            <Text size="sm" c="red">{run.error_message}</Text>
          </Paper>
        )}
      </Stack>
    );
  }

  function renderTesting(run: TestingRun) {
    const modelRun = trainingRunById.get(run.training_run_id);
    const pipeline = modelRun ? pipelineById.get(modelRun.training_pipeline_id) : undefined;
    const preprocessing = pipeline ? preprocessingById.get(pipeline.preprocessing_pipeline_id) : undefined;
    const geometry = run.roi_geometry as { x: number; y: number; width: number; height: number } | null;
    return (
      <Stack gap="sm">
        <Row label="Type">
          <Badge color="grape" variant="light">Inference</Badge>
        </Row>
        <Row label="Model">
          <Group gap="xs">
            <Text size="sm">{run.training_pipeline_name} · {run.method_type}</Text>
            <Badge size="sm" variant="light">{run.artifact_kind}</Badge>
          </Group>
        </Row>
        <Row label="Inference dataset"><Text size="sm">{run.training_dataset_name}</Text></Row>
        <Row label="Preprocessing"><PreprocessingSteps pipeline={preprocessing} /></Row>
        <Row label="ROI">
          {run.roi_name && geometry ? (
            <Text size="sm">{run.roi_name} · x {geometry.x}, y {geometry.y}, {geometry.width}x{geometry.height}</Text>
          ) : (
            <Text size="sm" c="dimmed">Full image</Text>
          )}
        </Row>
        <Row label="Scores">
          <Text size="sm">
            {run.image_count ?? 0} images · mean {run.score_mean?.toExponential(3) ?? '—'} · min{' '}
            {run.score_min?.toExponential(3) ?? '—'} · max {run.score_max?.toExponential(3) ?? '—'} ·{' '}
            {formatDuration(run.duration_seconds)} · {run.device ?? '—'}
          </Text>
        </Row>
        <Row label="Artifact">
          <Text size="xs" className="path-text">{run.artifact_kind}: {run.artifact_path}</Text>
        </Row>
        {run.error_message && (
          <Paper withBorder p="xs" radius="sm" bg="var(--mantine-color-red-0)">
            <Text size="sm" c="red">{run.error_message}</Text>
          </Paper>
        )}
      </Stack>
    );
  }

  return (
    <Modal opened={job !== null} onClose={onClose} title={title} size="xl">
      {job && (job.kind === 'train' ? renderTraining(job.run) : renderTesting(job.run))}
    </Modal>
  );
}
