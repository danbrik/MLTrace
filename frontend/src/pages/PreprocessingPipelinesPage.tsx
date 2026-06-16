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
  Switch,
  Table,
  Text,
  TextInput,
  Textarea,
  Title,
  Tooltip,
} from '@mantine/core';
import { useDebouncedValue } from '@mantine/hooks';
import { notifications } from '@mantine/notifications';

import { StepCard } from '../components/StepCard';
import { ArrowDown, ArrowUp, Eye, Info, Pencil, Plus, Save, Settings2, Trash2, Upload } from 'lucide-react';
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

type PipelineDesignResolution = {
  input_width: number | null;
  input_height: number | null;
  output_width: number | null;
  output_height: number | null;
};

const EMPTY_DESIGN_RESOLUTION: PipelineDesignResolution = {
  input_width: null,
  input_height: null,
  output_width: null,
  output_height: null,
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

function sizeLabel(width: number | null | undefined, height: number | null | undefined): string {
  return width && height ? `${width}x${height}` : 'n/a';
}

function folderResolutionCompactLabel(folder: DatasetFolder): string {
  const resolutions = Object.keys(folder.resolution_summary ?? {}).sort();
  return resolutions.length > 0 ? resolutions.join(', ') : 'resolution unknown';
}

function folderFileTypeLabel(folder: DatasetFolder): string {
  const extensions = Object.keys(folder.extension_summary ?? {}).sort();
  return extensions.length > 0 ? extensions.join(', ') : 'filetype unknown';
}

function firstResolution(folder: DatasetFolder): { width: number; height: number } | null {
  const [resolution] = Object.keys(folder.resolution_summary ?? {}).sort();
  const match = resolution?.match(/^(\d+)x(\d+)$/);
  if (!match) return null;
  return { width: Number(match[1]), height: Number(match[2]) };
}

function folderImageMetadataValue(folder: DatasetFolder, key: string): string | number | null {
  const value = folder.image_metadata?.[key];
  if (typeof value === 'string' || typeof value === 'number') return value;
  return null;
}

function folderMetadataLabel(folder: DatasetFolder): string {
  const metadata = folder.image_metadata;
  if (!metadata) return 'metadata unknown';
  const parts = [
    folderImageMetadataValue(folder, 'format') ? `format ${folderImageMetadataValue(folder, 'format')}` : null,
    folderImageMetadataValue(folder, 'mode') ? `mode ${folderImageMetadataValue(folder, 'mode')}` : null,
    folderImageMetadataValue(folder, 'dtype') ? `dtype ${folderImageMetadataValue(folder, 'dtype')}` : null,
    folderImageMetadataValue(folder, 'channels') ? `${folderImageMetadataValue(folder, 'channels')} ch` : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(', ') : 'metadata unknown';
}

function nextAvailablePipelineName(pipelines: PreprocessingPipeline[]): string {
  const used = new Set(pipelines.map((pipeline) => pipeline.name.trim().toLowerCase()));
  const base = 'Untitled pipeline';
  if (!used.has(base.toLowerCase())) return base;
  for (let index = 2; index < 10000; index += 1) {
    const candidate = `${base} ${index}`;
    if (!used.has(candidate.toLowerCase())) return candidate;
  }
  return `${base} ${Date.now()}`;
}

export function PreprocessingPipelinesPage() {
  const [steps, setSteps] = useState<PreprocessingStepDefinition[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [description, setDescription] = useState('');
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreprocessingPreview | null>(null);
  const [previewStale, setPreviewStale] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sourceImage, setSourceImage] = useState<PreprocessingPreviewImage | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [loadedPipelineId, setLoadedPipelineId] = useState<number | null>(null);
  const [isEditingLoadedPipeline, setIsEditingLoadedPipeline] = useState(true);
  const [designResolution, setDesignResolution] = useState<PipelineDesignResolution>(EMPTY_DESIGN_RESOLUTION);
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

  useEffect(() => {
    if (loadedPipelineId == null && !nameTouched) {
      setName(nextAvailablePipelineName(pipelines));
    }
  }, [pipelines, loadedPipelineId, nameTouched]);

  const stepByType = useMemo(() => new Map(steps.map((step) => [step.type, step])), [steps]);

  const folderOptions = useMemo<FolderOption[]>(
    () =>
      datasets.flatMap((dataset) =>
        dataset.folders.map((folder) => ({
          value: String(folder.id),
          label: `${dataset.name} / ${folder.relative_path} (${folderResolutionCompactLabel(folder)}, ${folderFileTypeLabel(folder)})`,
          dataset,
          folder,
        })),
      ),
    [datasets],
  );

  const selectedFolderOption = useMemo(
    () => folderOptions.find((option) => option.value === selectedFolderId) ?? null,
    [folderOptions, selectedFolderId],
  );

  // A pipeline name must be unique (case-insensitive), ignoring the pipeline currently loaded.
  const nameClash = useMemo(() => {
    const trimmed = name.trim().toLowerCase();
    if (!trimmed) return false;
    return pipelines.some(
      (pipeline) => pipeline.name.trim().toLowerCase() === trimmed && pipeline.id !== loadedPipelineId,
    );
  }, [name, pipelines, loadedPipelineId]);

  const createNameClash = useMemo(() => {
    const trimmed = name.trim().toLowerCase();
    if (!trimmed) return false;
    return pipelines.some((pipeline) => pipeline.name.trim().toLowerCase() === trimmed);
  }, [name, pipelines]);

  const loadedReadOnly = loadedPipelineId != null && !isEditingLoadedPipeline;

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
    if (loadedReadOnly) return;
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
    if (loadedReadOnly) return;
    setNodes((current) =>
      current.map((node) =>
        node.id === nodeIdToUpdate
          ? { ...node, data: { ...node.data, config: { ...node.data.config, ...partial } } }
          : node,
      ),
    );
    if (markStale) setPreviewStale(true);
  }

  function seedLoadImageFromFolder(folder: DatasetFolder, markStale = true) {
    const loadNode = nodes.find((node) => node.data.stepType === 'load_image');
    const resolution = firstResolution(folder);
    if (!loadNode || !resolution) return;
    updateNodeConfigMany(
      loadNode.id,
      {
        mode: 'unchanged',
        dtype: 'source',
        lock_size: true,
        lock_width: resolution.width,
        lock_height: resolution.height,
        source_format: folderImageMetadataValue(folder, 'format'),
        source_mode: folderImageMetadataValue(folder, 'mode'),
        source_dtype: folderImageMetadataValue(folder, 'dtype'),
        source_channels: folderImageMetadataValue(folder, 'channels'),
      },
      markStale,
    );
  }

  function addStep(step: PreprocessingStepDefinition) {
    if (loadedReadOnly) return;
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
    if (loadedReadOnly) return;
    const node = nodes.find((item) => item.id === nodeIdToRemove);
    if (!node || node.data.stepType === 'load_image') return;
    const nextNodes = nodes.filter((item) => item.id !== nodeIdToRemove);
    setNodes(nextNodes);
    setSelectedNodeId(nextNodes[0]?.id ?? null);
    setPreviewStale(true);
  }

  function moveNode(nodeIdToMove: string, direction: -1 | 1) {
    if (loadedReadOnly) return;
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

  function designResolutionFromPreview(): PipelineDesignResolution | null {
    const first = preview?.previews[0];
    const last = preview?.previews[preview.previews.length - 1];
    if (!first || !last) return null;
    return {
      input_width: first.width,
      input_height: first.height,
      output_width: last.width,
      output_height: last.height,
    };
  }

  function buildPipelinePayload() {
    const nextDesignResolution = designResolutionFromPreview();
    if (!nextDesignResolution) {
      notifications.show({
        color: 'yellow',
        title: 'Preview required',
        message: 'Run a preview before saving so MLTrace can store the design resolution.',
      });
      return null;
    }
    return {
      name,
      description,
      graph: backendGraph(),
      preview_folder_id: Number(selectedFolderId),
      ...nextDesignResolution,
    };
  }

  const designMismatchMessage = useMemo(() => {
    if (
      !sourceImage ||
      !designResolution.input_width ||
      !designResolution.input_height ||
      (sourceImage.width === designResolution.input_width && sourceImage.height === designResolution.input_height)
    ) {
      return null;
    }
    return `Pipeline tuned for ${designResolution.input_width}x${designResolution.input_height}, current preview image is ${sourceImage.width}x${sourceImage.height}.`;
  }, [sourceImage, designResolution]);

  const canStoreDesignResolution = Boolean(preview && !previewStale && !previewError);

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
    setNameTouched(true);
    setDescription(pipeline.description ?? '');
    setNodes(nextNodes);
    setSelectedNodeId(nextNodes[0]?.id ?? null);
    setPreview(null);
    setPreviewStale(false);
    setPreviewError(null);
    setLoadedPipelineId(pipeline.id);
    setIsEditingLoadedPipeline(false);
    setSelectedFolderId(pipeline.preview_folder_id ? String(pipeline.preview_folder_id) : null);
    setDesignResolution({
      input_width: pipeline.input_width,
      input_height: pipeline.input_height,
      output_width: pipeline.output_width,
      output_height: pipeline.output_height,
    });
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
    const payload = buildPipelinePayload();
    if (!payload) return;
    setLoading(true);
    try {
      const created = await createPreprocessingPipeline(payload);
      await refresh();
      setLoadedPipelineId(created.id);
      setIsEditingLoadedPipeline(false);
      setNameTouched(true);
      setDesignResolution({
        input_width: created.input_width,
        input_height: created.input_height,
        output_width: created.output_width,
        output_height: created.output_height,
      });
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
    const payload = buildPipelinePayload();
    if (!payload) return;
    setLoading(true);
    try {
      const updated = await updatePreprocessingPipeline(loadedPipelineId, payload);
      await refresh();
      setIsEditingLoadedPipeline(false);
      setDesignResolution({
        input_width: updated.input_width,
        input_height: updated.input_height,
        output_width: updated.output_width,
        output_height: updated.output_height,
      });
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
      if (!silent || message.includes('Input size is locked')) {
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
    setSourceImage(null);
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
      if (loadedPipelineId === pipeline.id) {
        setLoadedPipelineId(null);
        setIsEditingLoadedPipeline(true);
      }
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

  useEffect(() => {
    if (loadedReadOnly) return;
    if (!sourceImage) return;
    const loadNode = nodes.find((node) => node.data.stepType === 'load_image');
    if (!loadNode || loadNode.data.config.lock_size !== undefined) return;
    updateNodeConfigMany(
      loadNode.id,
      {
        lock_size: true,
        lock_width: sourceImage.width,
        lock_height: sourceImage.height,
      },
      true,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceImage, loadedReadOnly]);

  useEffect(() => {
    if (loadedReadOnly) return;
    if (!selectedFolderOption) return;
    seedLoadImageFromFolder(selectedFolderOption.folder);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFolderOption?.value, loadedReadOnly]);

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
          disabled={loadedReadOnly}
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
          disabled={loadedReadOnly}
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
          disabled={loadedReadOnly}
          onChange={(next) => updateNodeConfig(node.id, key, typeof next === 'number' ? next : property.default)}
        />
      );
    }

    return (
      <TextInput
        key={key}
        label={property.label ?? key}
        value={String(value)}
        disabled={loadedReadOnly}
        onChange={(event) => updateNodeConfig(node.id, key, event.currentTarget.value)}
      />
    );
  }

  function renderLoadImageSizeLock(node: PipelineNode) {
    const lockEnabled = node.data.config.lock_size === true;
    const lockWidth = typeof node.data.config.lock_width === 'number' ? node.data.config.lock_width : null;
    const lockHeight = typeof node.data.config.lock_height === 'number' ? node.data.config.lock_height : null;

    return (
      <Paper withBorder p="sm" radius="sm">
        <Group justify="space-between" align="flex-start">
          <div>
            <Text size="sm" fw={500}>
              Lock size
            </Text>
            <Text size="xs" c="dimmed">
              {lockEnabled
                ? `Locked to ${sizeLabel(lockWidth, lockHeight)}. Loading another image size will fail.`
                : sourceImage
                  ? `Current preview image is ${sourceImage.width}x${sourceImage.height}.`
                  : 'Select a preview folder before locking the input size.'}
            </Text>
          </div>
          <Switch
            checked={lockEnabled}
            disabled={loadedReadOnly || !sourceImage}
            onChange={(event) => {
              const checked = event.currentTarget.checked;
              if (checked && sourceImage) {
                updateNodeConfigMany(node.id, {
                  lock_size: true,
                  lock_width: sourceImage.width,
                  lock_height: sourceImage.height,
                });
              } else {
                updateNodeConfig(node.id, 'lock_size', false);
              }
            }}
          />
        </Group>
      </Paper>
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
        {node.data.stepType === 'load_image' && renderLoadImageSizeLock(node)}
        {fields.length > 0 && <SimpleGrid cols={{ base: 2, sm: 4 }}>{fields}</SimpleGrid>}
      </Stack>
    );
  }

  function renderSaveButtons() {
    if (loadedReadOnly) {
      return (
        <Button leftSection={<Pencil size={18} />} onClick={() => setIsEditingLoadedPipeline(true)}>
          Edit pipeline
        </Button>
      );
    }

    const createDisabled = !name.trim() || createNameClash || !canStoreDesignResolution;
    if (loadedPipelineId != null) {
      return (
        <>
          <Button
            leftSection={<Save size={18} />}
            onClick={handleUpdate}
            loading={loading}
            disabled={!name.trim() || nameClash || !canStoreDesignResolution}
          >
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

      <StepCard title="Pipeline details" color="blue">
          <TextInput
            label="Pipeline name"
            description={loadedReadOnly ? 'Loaded read-only.' : loadedPipelineId != null ? 'Editing a saved pipeline.' : undefined}
            value={name}
            disabled={loadedReadOnly}
            onChange={(event) => {
              setNameTouched(true);
              setName(event.currentTarget.value);
            }}
            error={nameClash ? 'A pipeline with this name already exists.' : undefined}
          />
          <Textarea label="Description" value={description} disabled={loadedReadOnly} onChange={(event) => setDescription(event.currentTarget.value)} />
          {(designResolution.input_width || designResolution.output_width) && (
            <Text size="xs" c="dimmed">
              Stored design input {sizeLabel(designResolution.input_width, designResolution.input_height)}, output{' '}
              {sizeLabel(designResolution.output_width, designResolution.output_height)}.
            </Text>
          )}
          {loadedReadOnly && (
            <Alert color="blue" title="Loaded read-only">
              Click Edit pipeline before changing this saved preprocessing pipeline.
            </Alert>
          )}
          {nameClash && (
            <Alert color="red" title="Name already exists">
              Choose a unique pipeline name before saving.
            </Alert>
          )}
          <Group justify="flex-end">
            {renderSaveButtons()}
          </Group>
      </StepCard>

      <StepCard index={1} title="Preview folder" color="violet" complete={Boolean(selectedFolderId && sourceImage)}>
          <Select
            label={
              <Group gap={6}>
                <Text component="span" size="sm" fw={500}>
                  Preview folder
                </Text>
                <Tooltip
                  multiline
                  w={320}
                  label="Choose a preview folder before editing the pipeline. MLTrace uses its first image to determine input size, seed size-dependent steps, and run previews."
                >
                  <ActionIcon size="xs" variant="subtle" aria-label="Preview folder info">
                    <Info size={14} />
                  </ActionIcon>
                </Tooltip>
              </Group>
            }
            description="Loaded once and reused by every preprocessing block."
            placeholder="Select a dataset folder"
            data={folderOptions.map((option) => ({ value: option.value, label: option.label }))}
            value={selectedFolderId}
            disabled={loadedReadOnly}
            onChange={(value) => {
              setSelectedFolderId(value);
              setSourceImage(null);
              setPreview(null);
              setPreviewStale(false);
              setPreviewError(null);
            }}
            searchable
          />
          {selectedFolderId && (
            <Stack gap={2}>
              <Text size="xs" c="dimmed">
                Folder: {selectedFolderOption ? `${folderResolutionCompactLabel(selectedFolderOption.folder)}, ${folderFileTypeLabel(selectedFolderOption.folder)}` : 'resolution unknown'}.
              </Text>
              {selectedFolderOption && (
                <Text size="xs" c="dimmed">
                  Dataset metadata: {folderMetadataLabel(selectedFolderOption.folder)}.
                </Text>
              )}
              <Text size="xs" c={sourceImage ? 'green' : 'dimmed'}>
                {sourceLoading
                  ? 'Loading preview image…'
                  : sourceImage
                    ? `Preview image loaded (${sourceImage.width}x${sourceImage.height}).`
                    : 'No preview image loaded yet.'}
              </Text>
            </Stack>
          )}
          {designMismatchMessage && (
            <Alert color="yellow" title="Design resolution mismatch">
              {designMismatchMessage}
            </Alert>
          )}
          <Group justify="flex-end">
            <Button leftSection={<Eye size={18} />} variant="light" onClick={handlePreview} loading={loading} disabled={!selectedFolderId}>
              Preview
            </Button>
          </Group>
      </StepCard>

      <StepCard index={2} title="Build pipeline" color="teal">
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
                    <ActionIcon
                      variant="light"
                      onClick={() => addStep(step)}
                      aria-label={`Add ${step.label}`}
                      disabled={loadedReadOnly || !selectedFolderId}
                    >
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
                const open = Boolean(selectedFolderId) && node.id === selectedNodeId;
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
                          <ActionIcon variant="subtle" disabled={loadedReadOnly || !selectedFolderId || index <= 1} onClick={() => moveNode(node.id, -1)}>
                            <ArrowUp size={16} />
                          </ActionIcon>
                          <ActionIcon
                            variant="subtle"
                            disabled={loadedReadOnly || !selectedFolderId || index === 0 || index === nodes.length - 1}
                            onClick={() => moveNode(node.id, 1)}
                          >
                            <ArrowDown size={16} />
                          </ActionIcon>
                          <ActionIcon
                            variant="subtle"
                            color="red"
                            disabled={loadedReadOnly || !selectedFolderId || node.data.stepType === 'load_image'}
                            onClick={() => removeNode(node.id)}
                          >
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
                        disabled={!selectedFolderId}
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
      </StepCard>

      <StepCard index={3} title="Preview" color="grape">
          <Group justify="space-between" align="flex-start">
            <div>
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
      </StepCard>

      <StepCard title="Saved pipelines" color="cyan">
          <ScrollArea>
            <Table striped verticalSpacing="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Steps</Table.Th>
                  <Table.Th>Design input</Table.Th>
                  <Table.Th>Design output</Table.Th>
                  <Table.Th>Description</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {pipelines.map((pipeline) => (
                  <Table.Tr key={pipeline.id}>
                    <Table.Td>{pipeline.name}</Table.Td>
                    <Table.Td>{pipeline.graph.nodes.length}</Table.Td>
                    <Table.Td>{sizeLabel(pipeline.input_width, pipeline.input_height)}</Table.Td>
                    <Table.Td>{sizeLabel(pipeline.output_width, pipeline.output_height)}</Table.Td>
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
      </StepCard>
    </Stack>
  );
}
