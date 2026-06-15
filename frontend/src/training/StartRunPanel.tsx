import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Paper,
  ScrollArea,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { ChevronDown, ChevronRight, Rocket, Search } from 'lucide-react';
import type { ReactNode } from 'react';
import { Fragment, useMemo, useState } from 'react';

import { formatValue, trainingModeColor, trainingModeLabel } from '../methods/utils';
import { orderedGraphNodes, stepDetail } from './graph';
import type { MethodConfiguration, PreprocessingPipeline, TrainingPipeline } from '../types';

// Method config keys worth surfacing in the expanded detail, per method family.
const METHOD_CONFIG_KEYS = [
  'input_channels',
  'input_width',
  'input_height',
  'latent_dim',
  'kl_weight',
  'output_activation',
  'aggregation',
  'accumulator_dtype',
  'output_dtype_policy',
];

function formatRes(width: number | null, height: number | null): string {
  return width && height ? `${width}x${height}` : 'n/a';
}

function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Group gap="sm" align="flex-start" wrap="nowrap">
      <Text size="xs" fw={700} tt="uppercase" c="dimmed" w={110} style={{ flexShrink: 0 }}>
        {label}
      </Text>
      <div style={{ flex: 1 }}>{children}</div>
    </Group>
  );
}

/**
 * Filterable overview of saved training pipelines. Each row is slim by default
 * and expands to reveal full details: datasets, image sizes, the actual
 * preprocessing steps, the method/architecture, and training parameters.
 */
export function StartRunPanel({
  pipelines,
  preprocessingPipelines,
  methodConfigurations,
  busyPipelineIds,
  onEnqueue,
  loading,
}: {
  pipelines: TrainingPipeline[];
  preprocessingPipelines: PreprocessingPipeline[];
  methodConfigurations: MethodConfiguration[];
  busyPipelineIds: Set<number>;
  onEnqueue: (pipelineId: number) => void;
  loading: boolean;
}) {
  const [search, setSearch] = useState('');
  const [methodFilter, setMethodFilter] = useState<string | null>(null);
  const [modeFilter, setModeFilter] = useState<string | null>(null);
  const [resolutionFilter, setResolutionFilter] = useState<string | null>(null);
  const [preprocessingFilter, setPreprocessingFilter] = useState<string | null>(null);
  const [datasetFilter, setDatasetFilter] = useState<string | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const preprocessingById = useMemo(
    () => new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline])),
    [preprocessingPipelines],
  );
  const methodById = useMemo(
    () => new Map(methodConfigurations.map((configuration) => [configuration.id, configuration])),
    [methodConfigurations],
  );

  const methodOptions = useMemo(
    () => [...new Set(pipelines.map((pipeline) => pipeline.method_type))].map((type) => ({ value: type, label: type })),
    [pipelines],
  );
  const resolutionOptions = useMemo(() => {
    const values = new Set<string>();
    pipelines.forEach((pipeline) => {
      const res = formatRes(pipeline.preprocessing_output_width, pipeline.preprocessing_output_height);
      if (res !== 'n/a') values.add(res);
    });
    return [...values].sort();
  }, [pipelines]);
  const preprocessingOptions = useMemo(
    () => [...new Set(pipelines.map((pipeline) => pipeline.preprocessing_pipeline_name))].sort(),
    [pipelines],
  );
  const datasetOptions = useMemo(() => {
    const values = new Set<string>();
    pipelines.forEach((pipeline) => pipeline.training_datasets.forEach((entry) => values.add(entry.name)));
    return [...values].sort();
  }, [pipelines]);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return pipelines.filter((pipeline) => {
      if (methodFilter && pipeline.method_type !== methodFilter) return false;
      if (modeFilter && pipeline.training_mode !== modeFilter) return false;
      if (
        resolutionFilter &&
        formatRes(pipeline.preprocessing_output_width, pipeline.preprocessing_output_height) !== resolutionFilter
      ) {
        return false;
      }
      if (preprocessingFilter && pipeline.preprocessing_pipeline_name !== preprocessingFilter) return false;
      if (datasetFilter && !pipeline.training_datasets.some((entry) => entry.name === datasetFilter)) return false;
      if (!query) return true;
      return (
        pipeline.name.toLowerCase().includes(query) ||
        pipeline.training_datasets.some((entry) => entry.name.toLowerCase().includes(query)) ||
        pipeline.method_configuration_name.toLowerCase().includes(query)
      );
    });
  }, [pipelines, search, methodFilter, modeFilter, resolutionFilter, preprocessingFilter, datasetFilter]);

  function toggleExpanded(id: number) {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <Paper withBorder p="md" radius="sm">
      <Stack gap="md">
        <Title order={3}>Start a Training Run</Title>
        {pipelines.length === 0 ? (
          <Alert color="blue">No training pipelines available yet. Create one on the Training Pipelines page.</Alert>
        ) : (
          <>
            <TextInput
              placeholder="Search by pipeline, method or dataset"
              leftSection={<Search size={16} />}
              value={search}
              onChange={(event) => setSearch(event.currentTarget.value)}
            />
            <Group grow>
              <Select placeholder="Method type" data={methodOptions} value={methodFilter} onChange={setMethodFilter} clearable />
              <Select
                placeholder="Training mode"
                data={[
                  { value: 'gradient', label: 'Gradient training' },
                  { value: 'fit', label: 'Training' },
                  { value: 'none', label: 'No Training' },
                ]}
                value={modeFilter}
                onChange={setModeFilter}
                clearable
              />
              <Select placeholder="Output size" data={resolutionOptions} value={resolutionFilter} onChange={setResolutionFilter} clearable />
              <Select placeholder="Preprocessing" data={preprocessingOptions} value={preprocessingFilter} onChange={setPreprocessingFilter} clearable />
              <Select placeholder="Dataset" data={datasetOptions} value={datasetFilter} onChange={setDatasetFilter} clearable searchable />
            </Group>
            <ScrollArea h={320}>
              <Table striped highlightOnHover verticalSpacing="sm">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th w={32} />
                    <Table.Th>Pipeline</Table.Th>
                    <Table.Th>Method</Table.Th>
                    <Table.Th>Images</Table.Th>
                    <Table.Th />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {filtered.map((pipeline) => {
                    const busy = busyPipelineIds.has(pipeline.id);
                    const expanded = expandedIds.has(pipeline.id);
                    const params = pipeline.training_parameters ?? {};
                    const preprocessing = preprocessingById.get(pipeline.preprocessing_pipeline_id);
                    const method = methodById.get(pipeline.method_configuration_id);
                    return (
                      <Fragment key={pipeline.id}>
                        <Table.Tr>
                          <Table.Td>
                            <ActionIcon variant="subtle" size="sm" onClick={() => toggleExpanded(pipeline.id)}>
                              {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                            </ActionIcon>
                          </Table.Td>
                          <Table.Td>{pipeline.name}</Table.Td>
                          <Table.Td>
                            <Group gap="xs" wrap="nowrap">
                              <Text size="sm">{pipeline.method_configuration_name}</Text>
                              <Badge color={trainingModeColor(pipeline.training_mode)} variant="light" size="sm">
                                {trainingModeLabel(pipeline.training_mode)}
                              </Badge>
                            </Group>
                          </Table.Td>
                          <Table.Td>{pipeline.total_selected_images}</Table.Td>
                          <Table.Td>
                            <Group justify="flex-end">
                              <Tooltip label={busy ? 'Already queued or running' : 'Enqueue training'}>
                                <Button
                                  size="compact-sm"
                                  leftSection={<Rocket size={14} />}
                                  variant="light"
                                  disabled={busy}
                                  loading={loading}
                                  onClick={() => onEnqueue(pipeline.id)}
                                >
                                  Enqueue
                                </Button>
                              </Tooltip>
                            </Group>
                          </Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                          <Table.Td colSpan={5} p={0} style={{ borderBottom: expanded ? undefined : 'none' }}>
                            <Collapse in={expanded}>
                              <Stack gap="sm" p="md" bg="var(--mantine-color-gray-0)">
                                {pipeline.description && (
                                  <Text size="sm" c="dimmed" fs="italic">
                                    {pipeline.description}
                                  </Text>
                                )}
                                <DetailRow label="Datasets">
                                  <Stack gap={2}>
                                    {pipeline.training_datasets.map((entry) => (
                                      <Text key={entry.training_dataset_id} size="sm">
                                        {entry.name}{' '}
                                        <Text span size="xs" c="dimmed">
                                          · {entry.total_selected_images} images · {entry.dataset_names.join(', ')}
                                        </Text>
                                      </Text>
                                    ))}
                                  </Stack>
                                </DetailRow>
                                <DetailRow label="Image size">
                                  <Group gap={6}>
                                    <Badge size="sm" variant="light" color="blue">
                                      in {formatRes(pipeline.preprocessing_input_width, pipeline.preprocessing_input_height)}
                                    </Badge>
                                    <Text size="xs" c="dimmed">
                                      →
                                    </Text>
                                    <Badge size="sm" variant="light" color="grape">
                                      out {formatRes(pipeline.preprocessing_output_width, pipeline.preprocessing_output_height)}
                                    </Badge>
                                  </Group>
                                </DetailRow>
                                <DetailRow label="Preprocessing">
                                  <Stack gap={4}>
                                    <Text size="sm">{pipeline.preprocessing_pipeline_name}</Text>
                                    {preprocessing ? (
                                      <Group gap={6}>
                                        {orderedGraphNodes(preprocessing).map((node, index) => (
                                          <Badge key={node.id} size="sm" variant="light" color="gray">
                                            {index + 1}. {node.type}
                                            <Text span size="xs" c="dimmed">
                                              {' '}
                                              ({stepDetail(node)})
                                            </Text>
                                          </Badge>
                                        ))}
                                      </Group>
                                    ) : (
                                      <Text size="xs" c="dimmed">
                                        Pipeline details unavailable.
                                      </Text>
                                    )}
                                  </Stack>
                                </DetailRow>
                                <DetailRow label="Method">
                                  <Stack gap={4}>
                                    <Group gap={6}>
                                      {METHOD_CONFIG_KEYS.filter((key) => method?.method_config?.[key] !== undefined).map(
                                        (key) => (
                                          <Badge key={key} size="sm" variant="light" color="indigo">
                                            {key}={formatValue(method?.method_config?.[key])}
                                          </Badge>
                                        ),
                                      )}
                                    </Group>
                                    {(method?.diagram?.nodes?.length ?? 0) > 0 && (
                                      <Group gap={6}>
                                        {method!.diagram.nodes.map((node, index) => (
                                          <Badge
                                            key={node.id}
                                            size="sm"
                                            variant="light"
                                            className={`model-section-${node.section}`}
                                          >
                                            {index + 1}. {node.label}
                                          </Badge>
                                        ))}
                                      </Group>
                                    )}
                                  </Stack>
                                </DetailRow>
                                <DetailRow label="Parameters">
                                  <Group gap={6}>
                                    <Badge size="sm" variant={pipeline.shuffle ? 'filled' : 'outline'} color="teal">
                                      {pipeline.shuffle ? 'shuffled' : 'in order'}
                                    </Badge>
                                    {Object.entries(params).map(([key, value]) => (
                                      <Badge key={key} size="sm" variant="light" color="gray">
                                        {key}={formatValue(value)}
                                      </Badge>
                                    ))}
                                    {Object.keys(params).length === 0 && (
                                      <Text size="xs" c="dimmed">
                                        no gradient parameters (fit-style method)
                                      </Text>
                                    )}
                                  </Group>
                                </DetailRow>
                              </Stack>
                            </Collapse>
                          </Table.Td>
                        </Table.Tr>
                      </Fragment>
                    );
                  })}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </>
        )}
        <Text size="xs" c="dimmed">
          Queued runs start automatically as GPUs become free; each run is pinned to one GPU (CPU fallback if none).
        </Text>
      </Stack>
    </Paper>
  );
}
