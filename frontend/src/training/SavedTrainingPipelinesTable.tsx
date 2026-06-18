import { ActionIcon, Alert, Badge, Group, Modal, Paper, ScrollArea, Stack, Table, Text, Title, Tooltip } from '@mantine/core';
import { Info, Trash2, Upload } from 'lucide-react';
import { useState } from 'react';
import type React from 'react';

import { formatValue } from '../methods/utils';
import { orderedGraphNodes, stepDetail } from './graph';
import type { MethodConfiguration, ModelLayerInstance, PreprocessingPipeline, TrainingDataset, TrainingPipeline } from '../types';

function keyTrainingParameters(pipeline: TrainingPipeline): string {
  const parameters = pipeline.training_parameters ?? {};
  const interesting = ['epochs', 'batch_size', 'learning_rate', 'loss', 'reconstruction_loss'];
  const parts = interesting
    .filter((key) => parameters[key] !== undefined)
    .map((key) => `${key} ${formatValue(parameters[key])}`);
  return parts.length > 0 ? parts.join(', ') : 'no training parameters';
}

function formatResolution(width: number | null | undefined, height: number | null | undefined): string {
  return width && height ? `${width}x${height}` : 'n/a';
}

function LayerList({ title, layers }: { title: string; layers?: ModelLayerInstance[] }) {
  if (!layers || layers.length === 0) return null;
  return (
    <Stack gap="xs">
      <Text fw={700} size="sm">{title}</Text>
      {layers.map((layer, index) => (
        <Paper key={layer.id} withBorder p="xs" radius="sm">
          <Text size="xs" fw={700}>{index + 1}. {layer.type}</Text>
          <Group gap={4} mt={4}>
            {Object.entries(layer.config ?? {}).map(([key, value]) => (
              <Badge key={key} size="xs" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      ))}
    </Stack>
  );
}

function renderTrainsets(pipeline: TrainingPipeline, datasets: TrainingDataset[]) {
  const datasetById = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  return (
    <Stack gap="md">
      {pipeline.training_datasets.map((entry) => {
        const dataset = datasetById.get(entry.training_dataset_id);
        return (
          <Paper key={entry.training_dataset_id} withBorder p="sm" radius="sm">
            <Stack gap="xs">
              <Group justify="space-between">
                <Text fw={700}>{entry.name}</Text>
                <Badge variant="light">
                  {dataset?.counts_missing ? 'Counts need refresh' : `${entry.total_selected_images} images`}
                </Badge>
              </Group>
              <Text size="xs" c="dimmed">
                Sources {entry.dataset_names.join(', ')}
              </Text>
              {dataset && (
                <Table striped verticalSpacing="xs">
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Dataset</Table.Th>
                      <Table.Th>Folder</Table.Th>
                      <Table.Th>Start</Table.Th>
                      <Table.Th>End</Table.Th>
                      <Table.Th>Stride</Table.Th>
                      <Table.Th>Images</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {dataset.rules.map((rule) => (
                      <Table.Tr key={rule.id}>
                        <Table.Td>{rule.dataset_name}</Table.Td>
                        <Table.Td>{rule.folder_relative_path}</Table.Td>
                        <Table.Td>{new Date(rule.start_timestamp).toLocaleString()}</Table.Td>
                        <Table.Td>{new Date(rule.end_timestamp).toLocaleString()}</Table.Td>
                        <Table.Td>{rule.stride}</Table.Td>
                        <Table.Td>{rule.selected_images == null ? 'Needs refresh' : rule.selected_images}</Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              )}
            </Stack>
          </Paper>
        );
      })}
    </Stack>
  );
}

function renderPreprocessing(pipeline: PreprocessingPipeline | undefined) {
  if (!pipeline) return <Alert color="yellow">Preprocessing details are unavailable.</Alert>;
  return (
    <Stack gap="sm">
      <Group>
        <Badge variant="light">Input {formatResolution(pipeline.input_width, pipeline.input_height)}</Badge>
        <Badge variant="light" color="yellow">Output {formatResolution(pipeline.output_width, pipeline.output_height)}</Badge>
      </Group>
      {orderedGraphNodes(pipeline).map((node, index) => (
        <Paper key={node.id} withBorder p="sm" radius="sm">
          <Text fw={700} size="sm">{index + 1}. {node.type}</Text>
          <Text size="xs" c="dimmed">{stepDetail(node)}</Text>
          <Group gap={6} mt={6}>
            {Object.entries(node.config ?? {}).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      ))}
    </Stack>
  );
}

function renderMethod(method: MethodConfiguration | undefined) {
  if (!method) return <Alert color="yellow">Method details are unavailable.</Alert>;
  return (
    <Stack gap="md">
      <div>
        <Text fw={700}>{method.name}</Text>
        {method.description && <Text size="sm" c="dimmed">{method.description}</Text>}
      </div>
      <Group gap={6}>
        {Object.entries(method.method_config ?? {}).map(([key, value]) => (
          <Badge key={key} size="sm" variant="light" color="gray">
            {key}={formatValue(value)}
          </Badge>
        ))}
      </Group>
      {method.method_graph.latent && (
        <Paper withBorder p="sm" radius="sm">
          <Text size="sm" fw={700}>Latent</Text>
          <Group gap={6} mt={6}>
            {Object.entries(method.method_graph.latent).map(([key, value]) => (
              <Badge key={key} size="sm" variant="light" color="gray">
                {key}={formatValue(value)}
              </Badge>
            ))}
          </Group>
        </Paper>
      )}
      <LayerList title="Encoder layers" layers={method.method_graph.encoder} />
      <LayerList title="Decoder layers" layers={method.method_graph.decoder} />
    </Stack>
  );
}

export function SavedTrainingPipelinesTable({
  pipelines,
  trainingDatasets = [],
  preprocessingPipelines = [],
  methodConfigurations = [],
  onLoad,
  onDelete,
}: {
  pipelines: TrainingPipeline[];
  trainingDatasets?: TrainingDataset[];
  preprocessingPipelines?: PreprocessingPipeline[];
  methodConfigurations?: MethodConfiguration[];
  onLoad: (pipelineId: number) => void;
  onDelete: (pipeline: TrainingPipeline) => void;
}) {
  const [detail, setDetail] = useState<{ title: string; body: React.ReactNode } | null>(null);
  const preprocessingById = new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline]));
  const methodById = new Map(methodConfigurations.map((method) => [method.id, method]));

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Title order={3}>Saved Training Pipelines</Title>
        <ScrollArea>
          <Table striped verticalSpacing="sm" miw={1080}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Trainsets</Table.Th>
                <Table.Th>Preprocessing</Table.Th>
                <Table.Th>Method</Table.Th>
                <Table.Th>Shuffle</Table.Th>
                <Table.Th>Key parameters</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {pipelines.map((pipeline) => (
                <Table.Tr key={pipeline.id}>
                  <Table.Td>{pipeline.name}</Table.Td>
                  <Table.Td>
                    <Stack gap={2}>
                      <Group gap={4} wrap="nowrap">
                        <Text size="sm">
                          {pipeline.training_datasets.map((entry) => entry.name).join(', ')}
                        </Text>
                        <Tooltip label="Inspect trainsets">
                          <ActionIcon
                            size="sm"
                            variant="subtle"
                            onClick={() => setDetail({ title: `Trainsets: ${pipeline.name}`, body: renderTrainsets(pipeline, trainingDatasets) })}
                          >
                            <Info size={14} />
                          </ActionIcon>
                        </Tooltip>
                      </Group>
                      <Text size="xs" c="dimmed">
                        {pipeline.total_selected_images} images total
                      </Text>
                    </Stack>
                  </Table.Td>
                  <Table.Td>
                    <Group gap={4} wrap="nowrap">
                      <Text size="sm">{pipeline.preprocessing_pipeline_name}</Text>
                      <Tooltip label="Inspect preprocessing">
                        <ActionIcon
                          size="sm"
                          variant="subtle"
                          onClick={() =>
                            setDetail({
                              title: `Preprocessing: ${pipeline.preprocessing_pipeline_name}`,
                              body: renderPreprocessing(preprocessingById.get(pipeline.preprocessing_pipeline_id)),
                            })
                          }
                        >
                          <Info size={14} />
                        </ActionIcon>
                      </Tooltip>
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Group gap={4} wrap="nowrap">
                      <Text size="sm">{pipeline.method_configuration_name}</Text>
                      <Tooltip label="Inspect method architecture">
                        <ActionIcon
                          size="sm"
                          variant="subtle"
                          onClick={() =>
                            setDetail({
                              title: `Method: ${pipeline.method_configuration_name}`,
                              body: renderMethod(methodById.get(pipeline.method_configuration_id)),
                            })
                          }
                        >
                          <Info size={14} />
                        </ActionIcon>
                      </Tooltip>
                    </Group>
                  </Table.Td>
                  <Table.Td>
                    <Badge variant={pipeline.shuffle ? 'filled' : 'outline'} color="teal" size="sm">
                      {pipeline.shuffle ? 'shuffled' : 'in order'}
                    </Badge>
                  </Table.Td>
                  <Table.Td>{keyTrainingParameters(pipeline)}</Table.Td>
                  <Table.Td>
                    <Group gap="xs" justify="flex-end" wrap="nowrap">
                      <Tooltip label="Load">
                        <ActionIcon variant="subtle" onClick={() => onLoad(pipeline.id)}>
                          <Upload size={18} />
                        </ActionIcon>
                      </Tooltip>
                      <Tooltip label="Delete">
                        <ActionIcon color="red" variant="subtle" onClick={() => onDelete(pipeline)}>
                          <Trash2 size={18} />
                        </ActionIcon>
                      </Tooltip>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        {pipelines.length === 0 && <Alert color="blue">Saved training pipelines will appear here.</Alert>}
        <Modal
          opened={detail !== null}
          onClose={() => setDetail(null)}
          title={detail?.title}
          size="xl"
          scrollAreaComponent={ScrollArea.Autosize}
        >
          {detail?.body}
        </Modal>
      </Stack>
    </Paper>
  );
}
