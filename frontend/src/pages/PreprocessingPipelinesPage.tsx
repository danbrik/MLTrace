import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Collapse,
  Divider,
  Grid,
  Group,
  NumberInput,
  Paper,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Textarea,
  Title,
  Tooltip,
} from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';
import { ArrowDown, ArrowUp, Eye, Plus, Save, Settings2, Trash2, Upload } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import {
  createPreprocessingPipeline,
  deletePreprocessingPipeline,
  getPreprocessingPipeline,
  listDatasets,
  listPreprocessingPipelines,
  listPreprocessingSteps,
  previewPreprocessingPipeline,
  updatePreprocessingPipeline,
} from '../api';
import { CONTROL_REGISTRY } from '../preprocessing/controls';
import type {
  Dataset,
  DatasetFolder,
  PreprocessingGraph,
  PreprocessingPipeline,
  PreprocessingPreview,
  PreprocessingPreviewImage,
  PreprocessingStepDefinition,
} from '../types';

type PipelineNodeData = {
  label: string;
  stepType: string;
  config: Record<string, unknown>;
};

type PipelineNode = {
  id: string;
  data: PipelineNodeData;
  position?: { x: number; y: number } | null;
};

type FolderOption = {
  value: string;
  label: string;
  dataset: Dataset;
  folder: DatasetFolder;
};

function nodeId(stepType: string): string {
  return `${stepType}-${crypto.randomUUID().slice(0, 8)}`;
}

function previewText(value: string | null): string {
  if (!value) return 'n/a';
  return value.replace('T', ' ');
}

function metadataLabel(item?: PreprocessingPreviewImage): string {
  if (!item) return 'not previewed';
  const color = item.channels === 1 ? 'grayscale' : `${item.channels} channels`;
  return `${item.width}x${item.height}, ${color}, ${item.dtype}`;
}

function ioLabel(kind: string | undefined, item?: PreprocessingPreviewImage): string {
  if (!item) return kind ?? 'not previewed';
  return `${kind ?? 'image ndarray'}: ${metadataLabel(item)}`;
}

export function PreprocessingPipelinesPage() {
  const [steps, setSteps] = useState<PreprocessingStepDefinition[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [name, setName] = useState('load_image to resize');
  const [description, setDescription] = useState('');
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>('load');
  const [preview, setPreview] = useState<PreprocessingPreview | null>(null);
  const [previewStale, setPreviewStale] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sourceImage, setSourceImage] = useState<PreprocessingPreviewImage | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [loadedPipelineId, setLoadedPipelineId] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [nodes, setNodes] = useState<PipelineNode[]>([
    {
      id: 'load',
      position: { x: 0, y: 0 },
      data: { label: 'Load image', stepType: 'load_image', config: {} },
    },
  ]);

  async function refresh() {
    const [nextSteps, nextDatasets, nextPipelines] = await Promise.all([
      listPreprocessingSteps(),
      listDatasets(),
      listPreprocessingPipelines(),
    ]);
    setSteps(nextSteps);
    setDatasets(nextDatasets.filter((dataset) => dataset.status === 'ready'));
    setPipelines(nextPipelines);
  }

  useEffect(() => {
    refresh().catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Could not load preprocessing data',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }, []);

  const stepByType = useMemo(() => new Map(steps.map((step) => [step.type, step])), [steps]);

  const folderOptions = useMemo<FolderOption[]>(
    () =>
      datasets.flatMap((dataset) =>
        dataset.folders.map((folder) => ({
          value: String(folder.id),
          label: `${dataset.name} / ${folder.relative_path} (${folder.image_count})`,
          dataset,
          folder,
        })),
      ),
    [datasets],
  );

  // A pipeline name must be unique (case-insensitive), ignoring the pipeline currently loaded.
  const nameClash = useMemo(() => {
    const trimmed = name.trim().toLowerCase();
    if (!trimmed) return false;
    return pipelines.some(
      (pipeline) => pipeline.name.trim().toLowerCase() === trimmed && pipeline.id !== loadedPipelineId,
    );
  }, [name, pipelines, loadedPipelineId]);

  // The image that flows INTO a step is the output of the preceding step. For the
  // first real step (index 1) the loaded source image is that output and is always
  // available immediately; deeper steps require a full pipeline preview to exist.
  function inputImageFor(index: number): PreprocessingPreviewImage | null {
    if (index <= 0) return null; // load_image has no image input
    const previousNode = nodes[index - 1];
    const fromPreview = preview?.previews.find((item) => item.node_id === previousNode.id) ?? null;
    if (fromPreview) return fromPreview;
    if (index === 1) return sourceImage;
    return null;
  }

  function stepLabel(stepType: string): string {
    return stepByType.get(stepType)?.label ?? stepType;
  }

  function updateNodeConfig(nodeIdToUpdate: string, key: string, value: unknown, markStale = true) {
    setNodes((current) =>
      current.map((node) =>
        node.id === nodeIdToUpdate
          ? { ...node, data: { ...node.data, config: { ...node.data.config, [key]: value } } }
          : node,
      ),
    );
    if (markStale) setPreviewStale(true);
  }

  function updateNodeConfigMany(nodeIdToUpdate: string, partial: Record<string, unknown>, markStale = true) {
    setNodes((current) =>
      current.map((node) =>
        node.id === nodeIdToUpdate
          ? { ...node, data: { ...node.data, config: { ...node.data.config, ...partial } } }
          : node,
      ),
    );
    if (markStale) setPreviewStale(true);
  }

  function addStep(step: PreprocessingStepDefinition) {
    if (step.type === 'load_image' && nodes.some((node) => node.data.stepType === 'load_image')) {
      notifications.show({ color: 'yellow', title: 'Only one load_image step is allowed', message: '' });
      return;
    }

    // Seed size-like fields from the previous step's actual output, so a new crop/resize/warp
    // adopts the current pixel count instead of a fixed 128x128.
    const inputImage = inputImageFor(nodes.length);
    const config: Record<string, unknown> = { ...step.default_config };
    if (inputImage) {
      for (const [key, property] of Object.entries(step.config_schema.properties)) {
        if (property.default_from === 'input_width') config[key] = inputImage.width;
        if (property.default_from === 'input_height') config[key] = inputImage.height;
      }
    }

    const nextNode: PipelineNode = {
      id: nodeId(step.type),
      position: { x: nodes.length * 220, y: 0 },
      data: {
        label: step.label,
        stepType: step.type,
        config,
      },
    };
    setNodes((current) => [...current, nextNode]);
    setSelectedNodeId(nextNode.id);
    setPreviewStale(true);
  }

  function removeNode(nodeIdToRemove: string) {
    const node = nodes.find((item) => item.id === nodeIdToRemove);
    if (!node || node.data.stepType === 'load_image') return;
    const nextNodes = nodes.filter((item) => item.id !== nodeIdToRemove);
    setNodes(nextNodes);
    setSelectedNodeId(nextNodes[0]?.id ?? null);
    setPreviewStale(true);
  }

  function moveNode(nodeIdToMove: string, direction: -1 | 1) {
    const index = nodes.findIndex((node) => node.id === nodeIdToMove);
    if (index < 0) return;
    const nextIndex = index + direction;
    if (nextIndex <= 0 || nextIndex >= nodes.length) return;
    const nextNodes = [...nodes];
    [nextNodes[index], nextNodes[nextIndex]] = [nextNodes[nextIndex], nextNodes[index]];
    setNodes(nextNodes);
    setPreviewStale(true);
  }

  function backendGraph(): PreprocessingGraph {
    return {
      nodes: nodes.map((node, index) => ({
        id: node.id,
        type: node.data.stepType,
        config: node.data.config,
        position: { x: index * 220, y: 0 },
      })),
      edges: nodes.slice(0, -1).map((node, index) => ({
        id: `${node.id}-${nodes[index + 1].id}`,
        source: node.id,
        target: nodes[index + 1].id,
      })),
    };
  }

  function loadGraph(pipeline: PreprocessingPipeline) {
    const nextNodes: PipelineNode[] = pipeline.graph.nodes.map((node, index) => ({
      id: node.id,
      position: node.position ?? { x: index * 220, y: 0 },
      data: {
        label: stepLabel(node.type),
        stepType: node.type,
        config: node.config ?? {},
      },
    }));
    setName(pipeline.name);
    setDescription(pipeline.description ?? '');
    setNodes(nextNodes);
    setSelectedNodeId(nextNodes[0]?.id ?? null);
    setPreview(null);
    setPreviewStale(false);
    setPreviewError(null);
    setLoadedPipelineId(pipeline.id);
  }

  async function handleLoadPipeline(pipelineId: number) {
    try {
      loadGraph(await getPreprocessingPipeline(pipelineId));
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Could not load pipeline',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  async function handleCreate() {
    setLoading(true);
    try {
      const created = await createPreprocessingPipeline({ name, description, graph: backendGraph() });
      await refresh();
      setLoadedPipelineId(created.id);
      notifications.show({ color: 'green', title: 'Pipeline saved', message: created.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Save failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleUpdate() {
    if (loadedPipelineId == null) return;
    setLoading(true);
    try {
      const updated = await updatePreprocessingPipeline(loadedPipelineId, { name, description, graph: backendGraph() });
      await refresh();
      notifications.show({ color: 'green', title: 'Pipeline updated', message: updated.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Update failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setLoading(false);
    }
  }

  async function runPreview(silent: boolean) {
    if (!selectedFolderId) return;
    setLoading(true);
    try {
      setPreview(await previewPreprocessingPipeline({ folder_id: Number(selectedFolderId), graph: backendGraph() }));
      setPreviewStale(false);
      setPreviewError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setPreviewError(message);
      if (!silent) {
        notifications.show({ color: 'red', title: 'Preview failed', message });
      }
    } finally {
      setLoading(false);
    }
  }

  async function handlePreview() {
    await runPreview(false);
  }

  async function loadSourceImage(folderId: string) {
    setSourceLoading(true);
    try {
      const result = await previewPreprocessingPipeline({
        folder_id: Number(folderId),
        graph: {
          nodes: [{ id: 'source-load', type: 'load_image', config: {}, position: { x: 0, y: 0 } }],
          edges: [],
        },
      });
      setSourceImage(result.previews[0] ?? null);
    } catch (error) {
      setSourceImage(null);
      notifications.show({
        color: 'red',
        title: 'Could not load preview image',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setSourceLoading(false);
    }
  }

  async function handleDelete(pipeline: PreprocessingPipeline) {
    if (!window.confirm(`Delete preprocessing pipeline "${pipeline.name}"?`)) return;
    try {
      await deletePreprocessingPipeline(pipeline.id);
      await refresh();
      notifications.show({ color: 'green', title: 'Pipeline deleted', message: pipeline.name });
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Delete failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  // Load the source image as soon as a preview folder is selected, and keep it
  // available for every preprocessing block regardless of full-pipeline previews.
  useEffect(() => {
    if (!selectedFolderId) {
      setSourceImage(null);
      return;
    }
    loadSourceImage(selectedFolderId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFolderId]);

  // Keep the full-pipeline preview in sync automatically (debounced) so every block's
  // input image (the previous step's output) always reflects the current configuration.
  const [debouncedNodes] = useDebouncedValue(nodes, 400);
  useEffect(() => {
    if (!selectedFolderId) return;
    runPreview(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedNodes, selectedFolderId]);

  // Generic input -> output preview shown inside every step block. The input image is the
  // previous step's output (inputImageFor), the output is this node's preview. Steps that
  // declare an interactive control (point_picker / crop) draw it on top of the input image.
  function renderStepPreview(node: PipelineNode, index: number) {
    const step = stepByType.get(node.data.stepType);
    if (!step) return null;
    const outputPreview = preview?.previews.find((item) => item.node_id === node.id) ?? null;
    const inputImage = inputImageFor(index);
    const isLoad = node.data.stepType === 'load_image';
    const controlId = step.config_schema.ui_control;
    const control = controlId ? CONTROL_REGISTRY[controlId] : undefined;

    let leftBody: ReactNode;
    if (isLoad) {
      leftBody = (
        <Text size="xs" c="dimmed">
          Loads the first image of the selected preview folder as the pipeline source.
        </Text>
      );
    } else if (!inputImage) {
      leftBody = <Alert color="blue">{sourceLoading ? 'Loading preview image…' : 'Run a preview to load the input image.'}</Alert>;
    } else if (control) {
      const Control = control.component;
      leftBody = (
        <Control
          inputImage={inputImage}
          config={node.data.config}
          onChange={(partial) => updateNodeConfigMany(node.id, partial)}
        />
      );
    } else {
      leftBody = <img src={inputImage.image_data_url} alt="Step input" className="preview-image" />;
    }

    return (
      <Stack gap="sm">
        <SimpleGrid cols={{ base: 1, md: 2 }}>
          <Stack gap="xs">
            <Group justify="space-between">
              <Text size="sm" fw={500}>
                Input
              </Text>
              {inputImage && (
                <Text size="xs" c="dimmed">
                  {metadataLabel(inputImage)}
                </Text>
              )}
            </Group>
            {leftBody}
          </Stack>
          <Stack gap="xs">
            <Group justify="space-between">
              <Text size="sm" fw={500}>
                Output
              </Text>
              {outputPreview && (
                <Text size="xs" c="dimmed">
                  {metadataLabel(outputPreview)}
                </Text>
              )}
            </Group>
            {outputPreview ? (
              <img src={outputPreview.image_data_url} alt="Step output" className="preview-image" />
            ) : (
              <Alert color="blue">{selectedFolderId ? 'Running preview…' : 'Select a preview folder to see the output.'}</Alert>
            )}
          </Stack>
        </SimpleGrid>
      </Stack>
    );
  }

  function renderConfigField(
    node: PipelineNode,
    key: string,
    property: PreprocessingStepDefinition['config_schema']['properties'][string],
  ) {
    const value = node.data.config[key] ?? property.default ?? '';

    if (property.enum) {
      return (
        <Select
          key={key}
          label={property.label ?? key}
          data={property.enum}
          value={String(value)}
          onChange={(next) => updateNodeConfig(node.id, key, next ?? property.default)}
        />
      );
    }

    if (property.type === 'integer' || property.type === 'number') {
      return (
        <NumberInput
          key={key}
          label={property.label ?? key}
          min={property.minimum}
          max={property.maximum}
          value={typeof value === 'number' ? value : Number(value)}
          onChange={(next) => updateNodeConfig(node.id, key, typeof next === 'number' ? next : property.default)}
        />
      );
    }

    return (
      <TextInput
        key={key}
        label={property.label ?? key}
        value={String(value)}
        onChange={(event) => updateNodeConfig(node.id, key, event.currentTarget.value)}
      />
    );
  }

  function renderNodeConfig(node: PipelineNode, index: number) {
    const step = stepByType.get(node.data.stepType);
    if (!step) return <Text c="dimmed">Unknown step.</Text>;
    const controlId = step.config_schema.ui_control;
    const ownedKeys = controlId ? CONTROL_REGISTRY[controlId]?.ownedKeys ?? [] : [];
    const fields = Object.entries(step.config_schema.properties)
      .filter(([key]) => !ownedKeys.includes(key))
      .map(([key, property]) => renderConfigField(node, key, property))
      .filter(Boolean);

    return (
      <Stack gap="sm">
        {renderStepPreview(node, index)}
        {fields.length > 0 && <SimpleGrid cols={{ base: 2, sm: 4 }}>{fields}</SimpleGrid>}
      </Stack>
    );
  }

  function renderSaveButtons() {
    const createDisabled = !name.trim() || nameClash;
    if (loadedPipelineId != null) {
      return (
        <>
          <Button leftSection={<Save size={18} />} onClick={handleUpdate} loading={loading} disabled={!name.trim() || nameClash}>
            Update pipeline
          </Button>
          <Button variant="light" leftSection={<Save size={18} />} onClick={handleCreate} loading={loading} disabled={createDisabled}>
            Save as new
          </Button>
        </>
      );
    }
    return (
      <Button leftSection={<Save size={18} />} onClick={handleCreate} loading={loading} disabled={createDisabled}>
        Save pipeline
      </Button>
    );
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Preprocessing</Title>
        <Text c="dimmed" size="sm">
          Build reusable linear image preprocessing pipelines and inspect each intermediate output.
        </Text>
      </div>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group grow align="flex-end">
            <TextInput
              label="Pipeline name"
              description={loadedPipelineId != null ? 'Editing a saved pipeline.' : undefined}
              value={name}
              onChange={(event) => setName(event.currentTarget.value)}
              error={nameClash ? 'A pipeline with this name already exists.' : undefined}
            />
            <Select
              label="Preview folder"
              description="Loaded once and reused by every preprocessing block."
              placeholder="Select a dataset folder"
              data={folderOptions.map((option) => ({ value: option.value, label: option.label }))}
              value={selectedFolderId}
              onChange={(value) => {
                setSelectedFolderId(value);
                setPreview(null);
                setPreviewStale(false);
              }}
              searchable
            />
          </Group>
          {selectedFolderId && (
            <Text size="xs" c={sourceImage ? 'green' : 'dimmed'}>
              {sourceLoading
                ? 'Loading preview image…'
                : sourceImage
                  ? `Preview image loaded (${sourceImage.width}x${sourceImage.height}).`
                  : 'No preview image loaded yet.'}
            </Text>
          )}
          <Textarea label="Description" value={description} onChange={(event) => setDescription(event.currentTarget.value)} />
          <Group justify="flex-end">
            <Button leftSection={<Eye size={18} />} variant="light" onClick={handlePreview} loading={loading} disabled={!selectedFolderId}>
              Preview
            </Button>
            {renderSaveButtons()}
          </Group>
        </Stack>
      </Paper>

      <Grid gutter="md" align="flex-start">
        <Grid.Col span={{ base: 12, lg: 3 }}>
          <Paper withBorder p="md" radius="sm">
            <Stack gap="sm">
              <Title order={3}>Step palette</Title>
              {steps.map((step) => (
                <Paper key={step.type} withBorder p="sm" radius="sm">
                  <Group justify="space-between" align="flex-start" wrap="nowrap">
                    <div>
                      <Text fw={700}>{step.label}</Text>
                      <Text size="xs" c="dimmed">
                        {step.input_kind}
                        {' -> '}
                        {step.output_kind}
                      </Text>
                      <Badge mt={6} size="xs" variant="light">
                        {step.category}
                      </Badge>
                    </div>
                    <ActionIcon variant="light" onClick={() => addStep(step)} aria-label={`Add ${step.label}`}>
                      <Plus size={16} />
                    </ActionIcon>
                  </Group>
                </Paper>
              ))}
            </Stack>
          </Paper>
        </Grid.Col>

        <Grid.Col span={{ base: 12, lg: 9 }}>
          <Paper withBorder p="md" radius="sm">
            <Stack gap="md">
              <Group justify="space-between">
                <Title order={3}>Pipeline order</Title>
                {previewStale && <Badge color="yellow">preview stale</Badge>}
              </Group>
              {previewError && (
                <Alert color="red" title="Pipeline invalid">
                  {previewError}
                </Alert>
              )}
              {nodes.map((node, index) => {
                const step = stepByType.get(node.data.stepType);
                const output = preview?.previews.find((item) => item.node_id === node.id);
                const input = index === 0 ? undefined : preview?.previews.find((item) => item.node_id === nodes[index - 1].id);
                const open = node.id === selectedNodeId;
                return (
                  <Paper key={node.id} withBorder p="sm" radius="sm" className={open ? 'pipeline-step selected' : 'pipeline-step'}>
                    <Stack gap="xs">
                      <Group justify="space-between" align="flex-start">
                        <Group gap="xs">
                          <Badge>{index + 1}</Badge>
                          <div>
                            <Text fw={700}>{node.data.label}</Text>
                            <Text size="xs" c="dimmed">
                              {node.data.stepType}
                            </Text>
                          </div>
                        </Group>
                        <Group gap={4}>
                          <ActionIcon variant="subtle" disabled={index <= 1} onClick={() => moveNode(node.id, -1)}>
                            <ArrowUp size={16} />
                          </ActionIcon>
                          <ActionIcon variant="subtle" disabled={index === 0 || index === nodes.length - 1} onClick={() => moveNode(node.id, 1)}>
                            <ArrowDown size={16} />
                          </ActionIcon>
                          <ActionIcon variant="subtle" color="red" disabled={node.data.stepType === 'load_image'} onClick={() => removeNode(node.id)}>
                            <Trash2 size={16} />
                          </ActionIcon>
                        </Group>
                      </Group>
                      <SimpleGrid cols={2}>
                        <div>
                          <Text size="xs" fw={600}>Input</Text>
                          <Text size="xs" c="dimmed">{index === 0 ? step?.input_kind ?? 'file' : ioLabel(step?.input_kind, input)}</Text>
                        </div>
                        <div>
                          <Text size="xs" fw={600}>Output</Text>
                          <Text size="xs" c="dimmed">{ioLabel(step?.output_kind, output)}</Text>
                        </div>
                      </SimpleGrid>
                      <Button
                        size="compact-sm"
                        variant={open ? 'filled' : 'light'}
                        leftSection={<Settings2 size={14} />}
                        onClick={() => setSelectedNodeId(open ? null : node.id)}
                      >
                        {open ? 'Close configuration' : 'Configure'}
                      </Button>
                      <Collapse in={open}>
                        <Divider my="xs" />
                        {renderNodeConfig(node, index)}
                      </Collapse>
                    </Stack>
                  </Paper>
                );
              })}
            </Stack>
          </Paper>
        </Grid.Col>
      </Grid>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group justify="space-between" align="flex-start">
            <div>
              <Title order={3}>Preview</Title>
              <Text size="sm" c="dimmed" className="path-text">
                {preview ? `${preview.source_image_path} at ${previewText(preview.source_timestamp)}` : 'Select a folder and run preview.'}
              </Text>
            </div>
            <Button leftSection={<Eye size={18} />} variant="light" onClick={handlePreview} loading={loading} disabled={!selectedFolderId}>
              Preview
            </Button>
          </Group>
          {preview ? (
            <SimpleGrid cols={{ base: 1, sm: 2, xl: 3 }}>
              {preview.previews.map((item, index) => (
                <Paper key={item.node_id} withBorder p="sm" radius="sm">
                  <Stack gap="xs">
                    <Group justify="space-between">
                      <Group gap="xs">
                        <Badge>{index + 1}</Badge>
                        <Text fw={600}>{item.label}</Text>
                      </Group>
                      <Badge variant="light">{item.width}x{item.height}</Badge>
                    </Group>
                    <img src={item.image_data_url} alt={item.label} className="preview-image" />
                    <Text size="xs" c="dimmed">
                      {item.dtype}, {item.channels} channel(s), min {item.value_min.toFixed(2)}, max {item.value_max.toFixed(2)}
                    </Text>
                  </Stack>
                </Paper>
              ))}
            </SimpleGrid>
          ) : (
            <Alert color="blue">Preview results will appear here in pipeline order.</Alert>
          )}
          <Divider />
          <Group justify="flex-end">
            <Button leftSection={<Eye size={18} />} variant="light" onClick={handlePreview} loading={loading} disabled={!selectedFolderId}>
              Preview
            </Button>
            {renderSaveButtons()}
          </Group>
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Saved pipelines</Title>
          <ScrollArea>
            <Table striped verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Steps</Table.Th>
                  <Table.Th>Updated</Table.Th>
                  <Table.Th>Description</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {pipelines.map((pipeline) => (
                  <Table.Tr key={pipeline.id}>
                    <Table.Td>{pipeline.name}</Table.Td>
                    <Table.Td>{pipeline.graph.nodes.length}</Table.Td>
                    <Table.Td>{previewText(pipeline.updated_at)}</Table.Td>
                    <Table.Td>{pipeline.description ?? ''}</Table.Td>
                    <Table.Td>
                      <Group gap="xs" justify="flex-end">
                        <Tooltip label="Load">
                          <ActionIcon variant="subtle" onClick={() => handleLoadPipeline(pipeline.id)}>
                            <Upload size={18} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Delete">
                          <ActionIcon color="red" variant="subtle" onClick={() => handleDelete(pipeline)}>
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
        </Stack>
      </Paper>
    </Stack>
  );
}
