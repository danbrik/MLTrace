import {
  Alert,
  Button,
  Group,
  Paper,
  Stack,
  Switch,
  Text,
  Textarea,
  TextInput,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Pencil, Rocket, RotateCcw, Save } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  ApiError,
  createTrainingPipeline,
  deleteTrainingPipeline,
  enqueueTrainingRun,
  listMethodConfigurations,
  listMethodDefinitions,
  listPreprocessingPipelines,
  listTrainingDatasets,
  listTrainingPipelines,
  resolveDuplicateTrainingPipeline,
  updateTrainingPipeline,
} from '../api';
import { StepCard } from '../components/StepCard';
import { usePendingIds } from '../hooks/usePendingIds';
import { SchemaForm } from '../methods/schema/SchemaForm';
import type { NumericDraftState } from '../methods/types';
import { schemaDefaults } from '../methods/utils';
import { datasetSizeSignature, pipelineOutputResolution } from '../training/graph';
import { DryRunPanel } from '../training/DryRunPanel';
import { MethodConfigurationPicker } from '../training/MethodConfigurationPicker';
import { PreprocessingPipelinePicker } from '../training/PreprocessingPipelinePicker';
import { SavedTrainingPipelinesTable } from '../training/SavedTrainingPipelinesTable';
import { TrainingDatasetPicker } from '../training/TrainingDatasetPicker';
import { TrainingPipelineFlow } from '../training/TrainingPipelineFlow';
import type {
  MethodConfiguration,
  MethodDefinition,
  PreprocessingPipeline,
  TrainingDataset,
  TrainingPipeline,
  TrainingPipelinePayload,
} from '../types';

function nextAvailablePipelineName(pipelines: TrainingPipeline[]): string {
  const used = new Set(pipelines.map((pipeline) => pipeline.name.trim().toLowerCase()));
  const base = 'Untitled training pipeline';
  if (!used.has(base.toLowerCase())) return base;
  for (let index = 2; index < 10000; index += 1) {
    const candidate = `${base} ${index}`;
    if (!used.has(candidate.toLowerCase())) return candidate;
  }
  return `${base} ${Date.now()}`;
}

export function TrainingPipelinesPage({
  active = true,
  onRunQueued,
}: {
  active?: boolean;
  onRunQueued?: () => void;
}) {
  const [trainingDatasets, setTrainingDatasets] = useState<TrainingDataset[]>([]);
  const [preprocessingPipelines, setPreprocessingPipelines] = useState<PreprocessingPipeline[]>([]);
  const [configurations, setConfigurations] = useState<MethodConfiguration[]>([]);
  const [methodDefinitions, setMethodDefinitions] = useState<MethodDefinition[]>([]);
  const [pipelines, setPipelines] = useState<TrainingPipeline[]>([]);

  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [description, setDescription] = useState('');
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>([]);
  const [selectedPipelineId, setSelectedPipelineId] = useState<number | null>(null);
  const [selectedConfigurationId, setSelectedConfigurationId] = useState<number | null>(null);
  const [shuffle, setShuffle] = useState(true);
  const [trainingParameters, setTrainingParameters] = useState<Record<string, unknown>>({});
  const [loadedPipelineId, setLoadedPipelineId] = useState<number | null>(null);
  const [isEditingLoadedPipeline, setIsEditingLoadedPipeline] = useState(true);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [numericDrafts, setNumericDrafts] = useState<Record<string, NumericDraftState>>({});
  const rowActions = usePendingIds();

  async function refresh() {
    const [nextDatasets, nextPipelines, nextConfigurations, nextDefinitions, nextTrainingPipelines] =
      await Promise.all([
        listTrainingDatasets(),
        listPreprocessingPipelines(),
        listMethodConfigurations(),
        listMethodDefinitions(),
        listTrainingPipelines(),
      ]);
    setTrainingDatasets(nextDatasets);
    setPreprocessingPipelines(nextPipelines);
    setConfigurations(nextConfigurations);
    setMethodDefinitions(nextDefinitions);
    setPipelines(nextTrainingPipelines);
  }

  // The app keeps every page mounted and only toggles visibility, so the
  // building blocks (datasets, pipelines, methods) created on other pages
  // would otherwise stay stale. Re-fetch whenever this page becomes active.
  useEffect(() => {
    if (!active) return;
    refresh().catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Could not load training pipeline data',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }, [active]);

  useEffect(() => {
    if (loadedPipelineId == null && !nameTouched) {
      setName(nextAvailablePipelineName(pipelines));
    }
  }, [pipelines, loadedPipelineId, nameTouched]);

  const methodByType = useMemo(
    () => new Map(methodDefinitions.map((definition) => [definition.type, definition])),
    [methodDefinitions],
  );
  const configurationById = useMemo(
    () => new Map(configurations.map((configuration) => [configuration.id, configuration])),
    [configurations],
  );
  const pipelineById = useMemo(
    () => new Map(preprocessingPipelines.map((pipeline) => [pipeline.id, pipeline])),
    [preprocessingPipelines],
  );
  const datasetById = useMemo(
    () => new Map(trainingDatasets.map((dataset) => [dataset.id, dataset])),
    [trainingDatasets],
  );

  const selectedConfiguration = selectedConfigurationId != null ? configurationById.get(selectedConfigurationId) : undefined;
  const selectedPipeline = selectedPipelineId != null ? pipelineById.get(selectedPipelineId) : undefined;
  const selectedDefinition = selectedConfiguration ? methodByType.get(selectedConfiguration.method_type) : undefined;
  const selectedDatasets = selectedDatasetIds
    .map((id) => datasetById.get(id))
    .filter((dataset): dataset is TrainingDataset => dataset !== undefined);

  const trainingSchema = selectedDefinition?.training_schema;
  const hasTrainingParameters = Object.keys(trainingSchema?.properties ?? {}).length > 0;

  // Cross-filter constraints propagated down the size chain:
  //   dataset image size -> pipeline input size -> (pipeline output -> method input).
  const requiredInputResolutions = useMemo(
    () => datasetSizeSignature(selectedDatasetIds, datasetById),
    [selectedDatasetIds, datasetById],
  );
  const pipelineOutput = selectedPipeline ? pipelineOutputResolution(selectedPipeline) : null;

  // Pre-flight warning: selected dataset image size vs. preprocessing input size.
  const datasetInputMismatch = useMemo(() => {
    if (!selectedPipeline || !requiredInputResolutions) return null;
    const pipelineInput =
      selectedPipeline.input_width && selectedPipeline.input_height
        ? `${selectedPipeline.input_width}x${selectedPipeline.input_height}`
        : null;
    if (!pipelineInput || requiredInputResolutions.includes(pipelineInput)) return null;
    return (
      `Selected datasets are ${requiredInputResolutions.join(', ')} but the preprocessing pipeline expects ` +
      `input ${pipelineInput}. The dummy test will fail until the sizes match.`
    );
  }, [selectedPipeline, requiredInputResolutions]);

  // Pre-flight resolution comparison: warns before the user even runs the dummy test.
  const resolutionMismatch = useMemo(() => {
    if (!selectedPipeline || !selectedConfiguration) return null;
    const config = selectedConfiguration.method_config ?? {};
    const methodWidth = Number(config.input_width);
    const methodHeight = Number(config.input_height);
    if (!selectedPipeline.output_width || !selectedPipeline.output_height) return null;
    if (!Number.isFinite(methodWidth) || !Number.isFinite(methodHeight)) return null;
    if (selectedPipeline.output_width === methodWidth && selectedPipeline.output_height === methodHeight) return null;
    return (
      `Preprocessing output is ${selectedPipeline.output_width}x${selectedPipeline.output_height} but the ` +
      `method expects ${methodWidth}x${methodHeight}. The dummy test will fail until the shapes match.`
    );
  }, [selectedPipeline, selectedConfiguration]);

  const invalidNumericDrafts = useMemo(
    () => Object.entries(numericDrafts).filter(([, draft]) => draft.dirty && !draft.valid),
    [numericDrafts],
  );

  const nameClash = useMemo(() => {
    const trimmed = name.trim().toLowerCase();
    if (!trimmed) return false;
    return pipelines.some((pipeline) => pipeline.name.trim().toLowerCase() === trimmed && pipeline.id !== loadedPipelineId);
  }, [name, pipelines, loadedPipelineId]);

  // Reinitialize training parameters whenever the selected method changes:
  // schema defaults overlaid with the saved configuration's training config.
  function handleConfigurationChange(configurationId: number | null) {
    setSelectedConfigurationId(configurationId);
    const configuration = configurationId != null ? configurationById.get(configurationId) : undefined;
    const definition = configuration ? methodByType.get(configuration.method_type) : undefined;
    setTrainingParameters({
      ...schemaDefaults(definition?.training_schema),
      ...(configuration?.training_config ?? {}),
    });
    setNumericDrafts({});
  }

  const payload: TrainingPipelinePayload | null = useMemo(() => {
    if (selectedDatasetIds.length === 0 || selectedPipelineId == null || selectedConfigurationId == null) {
      return null;
    }
    return {
      training_dataset_ids: selectedDatasetIds,
      preprocessing_pipeline_id: selectedPipelineId,
      method_configuration_id: selectedConfigurationId,
      shuffle,
      training_parameters: trainingParameters,
    };
  }, [selectedDatasetIds, selectedPipelineId, selectedConfigurationId, shuffle, trainingParameters]);

  function loadPipelineIntoBuilder(pipeline: TrainingPipeline) {
    setLoadedPipelineId(pipeline.id);
    setIsEditingLoadedPipeline(false);
    setName(pipeline.name);
    setNameTouched(true);
    setDescription(pipeline.description ?? '');
    setSelectedDatasetIds(pipeline.training_datasets.map((entry) => entry.training_dataset_id));
    setSelectedPipelineId(pipeline.preprocessing_pipeline_id);
    setSelectedConfigurationId(pipeline.method_configuration_id);
    setShuffle(pipeline.shuffle);
    setTrainingParameters(pipeline.training_parameters ?? {});
    setNumericDrafts({});
  }

  async function handleLoadPipeline(pipelineId: number) {
    await rowActions.runPending(`load:${pipelineId}`, async () => {
      const pipeline = pipelines.find((item) => item.id === pipelineId);
      if (pipeline) loadPipelineIntoBuilder(pipeline);
    });
  }

  function handleReset() {
    setLoadedPipelineId(null);
    setIsEditingLoadedPipeline(true);
    setNameTouched(false);
    setName(nextAvailablePipelineName(pipelines));
    setDescription('');
    setSelectedDatasetIds([]);
    setSelectedPipelineId(null);
    setSelectedConfigurationId(null);
    setShuffle(true);
    setTrainingParameters({});
    setNumericDrafts({});
  }

  function hasInvalidDrafts(): boolean {
    if (invalidNumericDrafts.length === 0) return false;
    notifications.show({
      color: 'red',
      title: 'Fix numeric inputs first',
      message: invalidNumericDrafts[0][1].message ?? 'One numeric input has an invalid draft value.',
    });
    return true;
  }

  async function handleSave(asNew: boolean) {
    if (!payload || hasInvalidDrafts()) return;
    setLoading(true);
    try {
      const savePayload = { ...payload, name: name.trim(), description };
      const saved =
        loadedPipelineId != null && !asNew
          ? await updateTrainingPipeline(loadedPipelineId, savePayload)
          : await createTrainingPipeline(savePayload);
      await refresh();
      loadPipelineIntoBuilder(saved);
      notifications.show({
        color: 'green',
        title: asNew || loadedPipelineId == null ? 'Training pipeline saved' : 'Training pipeline updated',
        message: saved.name,
      });
    } catch (error) {
      // 409 duplicate: name the existing pipeline (config-based, ignores name).
      if (error instanceof ApiError && error.status === 409) {
        notifications.show({ color: 'orange', title: 'Duplicate configuration', message: error.message });
      } else {
        notifications.show({
          color: 'red',
          title: 'Save failed',
          message: error instanceof Error ? error.message : 'Unknown error',
        });
      }
    } finally {
      setLoading(false);
    }
  }

  // "Run" = ensure a pipeline exists for the current configuration (reusing an
  // identical one if present, never creating a duplicate), then queue a run.
  async function handleRun() {
    if (!payload || hasInvalidDrafts()) return;
    setRunning(true);
    try {
      const { existing_pipeline } = await resolveDuplicateTrainingPipeline(payload);
      let target = existing_pipeline;
      if (target) {
        if (target.id !== loadedPipelineId) {
          loadPipelineIntoBuilder(target);
          notifications.show({
            color: 'blue',
            title: 'Using existing pipeline',
            message: `Identical configuration already saved as '${target.name}'.`,
          });
        }
      } else {
        target = await createTrainingPipeline({ ...payload, name: name.trim(), description });
        await refresh();
        loadPipelineIntoBuilder(target);
      }
      await enqueueTrainingRun(target.id);
      notifications.show({ color: 'green', title: 'Training queued', message: target.name });
      onRunQueued?.();
    } catch (error) {
      notifications.show({
        color: 'red',
        title: 'Run failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setRunning(false);
    }
  }

  async function handleDelete(pipeline: TrainingPipeline) {
    if (!window.confirm(`Delete training pipeline "${pipeline.name}"?`)) return;
    await rowActions.runPending(`delete:${pipeline.id}`, async () => {
      await deleteTrainingPipeline(pipeline.id);
      await refresh();
      if (loadedPipelineId === pipeline.id) handleReset();
      notifications.show({ color: 'green', title: 'Training pipeline deleted', message: pipeline.name });
    }).catch((error) => {
      notifications.show({
        color: 'red',
        title: 'Delete failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    });
  }

  const handleNumberDraftChange = useCallback((fieldId: string, state: NumericDraftState | null) => {
    setNumericDrafts((current) => {
      const next = { ...current };
      if (!state || !state.dirty) {
        delete next[fieldId];
      } else {
        next[fieldId] = state;
      }
      return next;
    });
  }, []);

  const saveDisabled = !payload || !name.trim() || nameClash || invalidNumericDrafts.length > 0;
  const loadedReadOnly = loadedPipelineId != null && !isEditingLoadedPipeline;
  const loadedPipeline = loadedPipelineId != null ? pipelines.find((pipeline) => pipeline.id === loadedPipelineId) ?? null : null;

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Training Pipelines</Title>
        <Text c="dimmed" size="sm">
          Compose training sets, a preprocessing pipeline, and a method into a saved training pipeline. The
          actual training run is triggered elsewhere.
        </Text>
      </div>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group grow align="flex-start">
            <TextInput
              label="Training pipeline name"
              value={name}
              disabled={loadedReadOnly}
              onChange={(event) => {
                setNameTouched(true);
                setName(event.currentTarget.value);
              }}
              error={nameClash ? 'A training pipeline with this name already exists.' : undefined}
            />
            <Textarea label="Description" rows={1} value={description} disabled={loadedReadOnly} onChange={(event) => setDescription(event.currentTarget.value)} />
          </Group>
          {loadedReadOnly && (
            <Alert color={loadedPipeline?.is_update_locked ? 'yellow' : 'blue'} title="Loaded read-only">
              <Stack gap={4}>
                <Text size="sm">Click Edit before changing this saved training pipeline. You can still run the loaded pipeline.</Text>
                {loadedPipeline?.update_lock_reasons.map((reason) => (
                  <Text key={reason} size="sm">
                    {reason}
                  </Text>
                ))}
              </Stack>
            </Alert>
          )}
          <Group justify="flex-end">
            <Button variant="default" leftSection={<RotateCcw size={18} />} onClick={handleReset}>
              Reset
            </Button>
            {loadedReadOnly ? (
              <Button leftSection={<Pencil size={18} />} onClick={() => setIsEditingLoadedPipeline(true)} disabled={loadedPipeline?.is_update_locked}>
                Edit pipeline
              </Button>
            ) : (
              <>
                {loadedPipelineId != null && (
                  <Button
                    variant="light"
                    leftSection={<Save size={18} />}
                    loading={loading}
                    disabled={saveDisabled || running}
                    onClick={() => handleSave(true)}
                  >
                    Save as New
                  </Button>
                )}
                <Button
                  variant="light"
                  leftSection={<Save size={18} />}
                  loading={loading}
                  disabled={saveDisabled || running}
                  onClick={() => handleSave(false)}
                >
                  {loadedPipelineId != null ? 'Update' : 'Save'}
                </Button>
              </>
            )}
            <Button
              leftSection={<Rocket size={18} />}
              loading={running}
              disabled={saveDisabled || loading}
              onClick={handleRun}
            >
              Run
            </Button>
          </Group>
        </Stack>
      </Paper>

      <StepCard index={1} title="Train/test sets" color="blue" complete={selectedDatasetIds.length > 0}>
        <TrainingDatasetPicker
          trainingDatasets={trainingDatasets}
          selectedIds={selectedDatasetIds}
          onChange={setSelectedDatasetIds}
          disabled={loadedReadOnly}
          embedded
        />
      </StepCard>

      <StepCard index={2} title="Preprocessing pipeline" color="violet" complete={selectedPipelineId != null}>
        <PreprocessingPipelinePicker
          pipelines={preprocessingPipelines}
          selectedId={selectedPipelineId}
          onChange={setSelectedPipelineId}
          requiredInputResolutions={requiredInputResolutions}
          disabled={loadedReadOnly}
          embedded
        />
      </StepCard>

      <StepCard index={3} title="Method / architecture" color="teal" complete={selectedConfigurationId != null}>
        <MethodConfigurationPicker
          configurations={configurations}
          methodByType={methodByType}
          selectedId={selectedConfigurationId}
          onChange={handleConfigurationChange}
          requiredInputResolution={pipelineOutput}
          disabled={loadedReadOnly}
          embedded
        />
      </StepCard>

      <StepCard index={4} title="Training parameters" color="grape">
        {!selectedConfiguration && (
          <Alert color="blue">Select a method to configure its training parameters.</Alert>
        )}
        {selectedConfiguration && !hasTrainingParameters && (
          <Alert color="blue">This method is fitted directly and has no gradient training parameters.</Alert>
        )}
        {selectedConfiguration && hasTrainingParameters && trainingSchema && (
          <SchemaForm
            schema={trainingSchema}
            config={trainingParameters}
            disabled={loadedReadOnly}
            fieldPrefix="training"
            onChange={(key, value) => setTrainingParameters((current) => ({ ...current, [key]: value }))}
            onNumberDraftChange={handleNumberDraftChange}
          />
        )}
        <Switch
          label="Shuffle combined training sets during training"
          checked={shuffle}
          disabled={loadedReadOnly}
          onChange={(event) => setShuffle(event.currentTarget.checked)}
        />
        {datasetInputMismatch && (
          <Alert color="yellow" title="Shape mismatch">
            {datasetInputMismatch}
          </Alert>
        )}
        {resolutionMismatch && (
          <Alert color="yellow" title="Shape mismatch">
            {resolutionMismatch}
          </Alert>
        )}
      </StepCard>

      <StepCard index={5} title="Overview & dry run" color="orange">
        <TrainingPipelineFlow
          trainingDatasets={selectedDatasets}
          shuffle={shuffle}
          preprocessingPipeline={selectedPipeline ?? null}
          configuration={selectedConfiguration ?? null}
        />
        <DryRunPanel payload={payload} disabled={!payload || invalidNumericDrafts.length > 0} />
      </StepCard>

      <SavedTrainingPipelinesTable
        pipelines={pipelines}
        trainingDatasets={trainingDatasets}
        preprocessingPipelines={preprocessingPipelines}
        methodConfigurations={configurations}
        onLoad={handleLoadPipeline}
        onDelete={handleDelete}
        isLoading={(pipelineId) => rowActions.isPending(`load:${pipelineId}`)}
        isDeleting={(pipelineId) => rowActions.isPending(`delete:${pipelineId}`)}
      />
    </Stack>
  );
}
