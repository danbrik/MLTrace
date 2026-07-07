import {
  Alert,
  Badge,
  Button,
  Group,
  MultiSelect,
  NumberInput,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Textarea,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Pause, Play, Plus, RotateCcw, Scissors, Square, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  abortOptimizationStudy,
  createOptimizationSplit,
  createOptimizationStudy,
  deleteOptimizationStudy,
  listMethodConfigurations,
  listOptimizationStudies,
  listPreprocessingPipelines,
  listTrainingDatasets,
  pauseOptimizationStudy,
  resumeOptimizationStudy,
  startOptimizationStudy,
} from '../api';
import { PlotlyChart } from '../components/PlotlyChart';
import type { Data, Layout } from '../lib/plotly';
import type {
  MethodConfiguration,
  OptimizationParameterSpec,
  OptimizationStudy,
  OptimizationStudyPayload,
  PreprocessingPipeline,
  TrainingDataset,
} from '../types';

const DEFAULT_SEARCH_SPACE: OptimizationParameterSpec[] = [
  { path: 'method_config.latent_dim', kind: 'categorical', choices: [64, 128, 200, 300, 512] },
  { path: 'method_config.bottleneck_channels', kind: 'categorical', choices: [4, 8, 16, 32, 64] },
  { path: 'training_parameters.learning_rate', kind: 'float', low: 0.00001, high: 0.001, log: true },
  { path: 'training_parameters.weight_decay', kind: 'float', low: 0.0, high: 0.0001 },
  { path: 'training_parameters.batch_size', kind: 'categorical', choices: [8, 16, 32] },
];

function statusColor(status: string): string {
  if (status === 'finished') return 'green';
  if (status === 'failed' || status === 'aborted') return 'red';
  if (status === 'running') return 'blue';
  if (status === 'paused') return 'yellow';
  return 'gray';
}

function parseChoices(value: string): Array<number | string> {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => {
      const number = Number(item);
      return Number.isFinite(number) && item !== '' ? number : item;
    });
}

function choiceText(spec: OptimizationParameterSpec): string {
  if (spec.kind === 'categorical') return (spec.choices ?? []).join(', ');
  return '';
}

function metricNumber(metrics: Record<string, unknown> | null, key: string): number | null {
  const value = metrics?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function objectivePlot(study: OptimizationStudy | null): { data: Data[]; layout: Partial<Layout> } {
  const trials = study?.trials.filter((trial) => typeof trial.objective_value === 'number') ?? [];
  return {
    data: [
      {
        type: 'scatter',
        mode: 'lines+markers',
        x: trials.map((trial) => trial.number),
        y: trials.map((trial) => trial.objective_value),
        name: 'Objective',
      } as Data,
    ],
    layout: {
      height: 260,
      margin: { t: 20, r: 20, b: 45, l: 55 },
      xaxis: { title: { text: 'Trial' } },
      yaxis: { title: { text: study?.objective_name ?? 'Objective' } },
    },
  };
}

function distributionPlot(study: OptimizationStudy | null): { data: Data[]; layout: Partial<Layout> } {
  const trials = study?.trials.filter((trial) => trial.metrics) ?? [];
  return {
    data: [
      {
        type: 'scatter',
        mode: 'markers',
        x: trials.map((trial) => trial.number),
        y: trials.map((trial) => metricNumber(trial.metrics, 'normal_p95')),
        name: 'Normal p95',
      } as Data,
      {
        type: 'scatter',
        mode: 'markers',
        x: trials.map((trial) => trial.number),
        y: trials.map((trial) => metricNumber(trial.metrics, 'anomaly_median')),
        name: 'Anomaly median',
      } as Data,
    ],
    layout: {
      height: 260,
      margin: { t: 20, r: 20, b: 45, l: 55 },
      xaxis: { title: { text: 'Trial' } },
      yaxis: { title: { text: 'Score' } },
    },
  };
}

function parameterSummaryPlot(study: OptimizationStudy | null): { data: Data[]; layout: Partial<Layout> } {
  const finished = study?.trials.filter((trial) => trial.status === 'finished' && typeof trial.objective_value === 'number') ?? [];
  const paths = Array.from(new Set(finished.flatMap((trial) => Object.keys(trial.sampled_params ?? {})))).filter(
    (path) => path !== 'method_configuration_id',
  );
  const values = paths.map((path) => {
    const numeric = finished
      .map((trial) => trial.sampled_params[path])
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
    if (numeric.length === 0) return 0;
    return new Set(numeric.map((value) => String(value))).size;
  });
  return {
    data: [{ type: 'bar', x: paths, y: values, name: 'Sampled value count' } as Data],
    layout: {
      height: 260,
      margin: { t: 20, r: 20, b: 90, l: 45 },
      xaxis: { tickangle: -30 },
      yaxis: { title: { text: 'Distinct sampled values' } },
    },
  };
}

export function OptimizationPage({ active = true }: { active?: boolean }) {
  const [studies, setStudies] = useState<OptimizationStudy[]>([]);
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [methods, setMethods] = useState<MethodConfiguration[]>([]);
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState<string | null>(null);

  const [name, setName] = useState('Optimization Study');
  const [description, setDescription] = useState('');
  const [preprocessingPipelineId, setPreprocessingPipelineId] = useState<string | null>(null);
  const [methodIds, setMethodIds] = useState<string[]>([]);
  const [normalTrainId, setNormalTrainId] = useState<string | null>(null);
  const [normalValidationId, setNormalValidationId] = useState<string | null>(null);
  const [normalHoldoutId, setNormalHoldoutId] = useState<string | null>(null);
  const [anomalyValidationId, setAnomalyValidationId] = useState<string | null>(null);
  const [anomalyHoldoutId, setAnomalyHoldoutId] = useState<string | null>(null);
  const [objectiveName, setObjectiveName] = useState('median_anomaly_minus_p95_normal');
  const [nTrials, setNTrials] = useState(10);
  const [maxParallelTrials, setMaxParallelTrials] = useState(1);
  const [sampler, setSampler] = useState<'tpe' | 'random'>('tpe');
  const [searchSpace, setSearchSpace] = useState<OptimizationParameterSpec[]>(DEFAULT_SEARCH_SPACE);

  const [splitPrefix, setSplitPrefix] = useState('Optuna split');
  const [splitNormalSource, setSplitNormalSource] = useState<string | null>(null);
  const [splitAnomalySource, setSplitAnomalySource] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [nextStudies, nextDatasets, nextPipelines, nextMethods] = await Promise.all([
      listOptimizationStudies(),
      listTrainingDatasets(),
      listPreprocessingPipelines(),
      listMethodConfigurations(),
    ]);
    setStudies(nextStudies);
    setDatasets(nextDatasets);
    setPipelines(nextPipelines);
    setMethods(nextMethods.filter((method) => method.supports_training_pipeline));
    if (!selectedStudyId && nextStudies[0]) setSelectedStudyId(String(nextStudies[0].id));
  }, [selectedStudyId]);

  useEffect(() => {
    if (!active) return;
    setLoading(true);
    refresh()
      .catch((error) => notifications.show({ color: 'red', title: 'Could not load optimization data', message: String(error) }))
      .finally(() => setLoading(false));
  }, [active, refresh]);

  useEffect(() => {
    if (!active) return;
    const handle = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(handle);
  }, [active, refresh]);

  const selectedStudy = useMemo(
    () => studies.find((study) => String(study.id) === selectedStudyId) ?? null,
    [selectedStudyId, studies],
  );

  const datasetOptions = datasets.map((dataset) => ({
    value: String(dataset.id),
    label: `${dataset.name} (${dataset.total_selected_images} images)`,
  }));
  const methodOptions = methods.map((method) => ({
    value: String(method.id),
    label: `${method.name} (${method.method_type})`,
  }));
  const preprocessingOptions = pipelines.map((pipeline) => ({
    value: String(pipeline.id),
    label: `${pipeline.name} (${pipeline.output_width ?? '?'}x${pipeline.output_height ?? '?'})`,
  }));

  function buildPayload(): OptimizationStudyPayload | null {
    if (!preprocessingPipelineId || methodIds.length === 0 || !normalTrainId || !normalValidationId || !anomalyValidationId) {
      notifications.show({ color: 'red', title: 'Study incomplete', message: 'Select preprocessing, methods, normal train, normal validation, and anomaly validation.' });
      return null;
    }
    return {
      name: name.trim(),
      description: description.trim() || null,
      preprocessing_pipeline_id: Number(preprocessingPipelineId),
      method_configuration_ids: methodIds.map(Number),
      normal_train_dataset_id: Number(normalTrainId),
      normal_validation_dataset_id: Number(normalValidationId),
      anomaly_validation_dataset_id: Number(anomalyValidationId),
      normal_holdout_dataset_id: normalHoldoutId ? Number(normalHoldoutId) : null,
      anomaly_holdout_dataset_id: anomalyHoldoutId ? Number(anomalyHoldoutId) : null,
      search_space: searchSpace.filter((spec) => spec.path.trim()),
      objective_name: objectiveName,
      direction: objectiveName === 'normal_validation_loss' ? 'minimize' : 'maximize',
      n_trials: nTrials,
      max_parallel_trials: maxParallelTrials,
      sampler,
      split_config: {},
      objective_config: {},
    };
  }

  async function handleCreateStudy() {
    const payload = buildPayload();
    if (!payload) return;
    setPending('create');
    try {
      const study = await createOptimizationStudy(payload);
      setSelectedStudyId(String(study.id));
      await refresh();
      notifications.show({ color: 'green', title: 'Study created', message: study.name });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Could not create study', message: error instanceof Error ? error.message : String(error) });
    } finally {
      setPending(null);
    }
  }

  async function handleSplit() {
    if (!splitNormalSource || !splitAnomalySource) {
      notifications.show({ color: 'red', title: 'Split incomplete', message: 'Select normal and anomaly source datasets.' });
      return;
    }
    setPending('split');
    try {
      const split = await createOptimizationSplit({
        name_prefix: splitPrefix,
        normal_source_dataset_id: Number(splitNormalSource),
        anomaly_source_dataset_id: Number(splitAnomalySource),
        normal_train_fraction: 0.75,
        normal_validation_fraction: 0.125,
        anomaly_validation_fraction: 0.5,
      });
      setNormalTrainId(String(split.normal_train_dataset.id));
      setNormalValidationId(String(split.normal_validation_dataset.id));
      setNormalHoldoutId(String(split.normal_holdout_dataset.id));
      setAnomalyValidationId(String(split.anomaly_validation_dataset.id));
      setAnomalyHoldoutId(String(split.anomaly_holdout_dataset.id));
      await refresh();
      notifications.show({ color: 'green', title: 'Split created', message: 'Train/validation/holdout datasets were created.' });
    } catch (error) {
      notifications.show({ color: 'red', title: 'Could not create split', message: error instanceof Error ? error.message : String(error) });
    } finally {
      setPending(null);
    }
  }

  async function runStudyAction(action: 'start' | 'pause' | 'resume' | 'abort' | 'delete') {
    if (!selectedStudy) return;
    setPending(action);
    try {
      if (action === 'start') await startOptimizationStudy(selectedStudy.id);
      if (action === 'pause') await pauseOptimizationStudy(selectedStudy.id);
      if (action === 'resume') await resumeOptimizationStudy(selectedStudy.id);
      if (action === 'abort') await abortOptimizationStudy(selectedStudy.id);
      if (action === 'delete') {
        await deleteOptimizationStudy(selectedStudy.id);
        setSelectedStudyId(null);
      }
      await refresh();
    } catch (error) {
      notifications.show({ color: 'red', title: `Could not ${action} study`, message: error instanceof Error ? error.message : String(error) });
    } finally {
      setPending(null);
    }
  }

  const objective = objectivePlot(selectedStudy);
  const distributions = distributionPlot(selectedStudy);
  const parameters = parameterSummaryPlot(selectedStudy);

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="flex-start">
        <div>
          <Title order={2}>Optimization</Title>
          <Text c="dimmed">Optuna-style hyperparameter studies using MLTrace Training and Inference runs.</Text>
        </div>
        {loading && <Badge variant="light">Loading</Badge>}
      </Group>

      <Paper withBorder p="md" radius="sm">
        <Stack>
          <Title order={3}>Create Study</Title>
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            <TextInput label="Study name" value={name} onChange={(event) => setName(event.currentTarget.value)} />
            <Select label="Objective" value={objectiveName} onChange={(value) => setObjectiveName(value ?? 'median_anomaly_minus_p95_normal')} data={[
              { value: 'median_anomaly_minus_p95_normal', label: 'Median anomaly - p95 normal' },
              { value: 'mean_gap', label: 'Mean anomaly - mean normal' },
              { value: 'roc_auc', label: 'ROC AUC' },
              { value: 'pr_auc', label: 'PR AUC' },
              { value: 'normal_validation_loss', label: 'Normal validation loss' },
            ]} />
            <Select label="Preprocessing pipeline" value={preprocessingPipelineId} onChange={setPreprocessingPipelineId} data={preprocessingOptions} searchable />
            <MultiSelect label="Methods to search" value={methodIds} onChange={setMethodIds} data={methodOptions} searchable />
          </SimpleGrid>
          <Textarea label="Description" value={description} onChange={(event) => setDescription(event.currentTarget.value)} autosize minRows={2} />

          <Paper withBorder p="sm" radius="sm">
            <Stack gap="sm">
              <Group justify="space-between">
                <Text fw={700}>Chronological split helper</Text>
                <Button leftSection={<Scissors size={16} />} variant="light" loading={pending === 'split'} onClick={handleSplit}>
                  Create split
                </Button>
              </Group>
              <SimpleGrid cols={{ base: 1, md: 3 }}>
                <TextInput label="Split prefix" value={splitPrefix} onChange={(event) => setSplitPrefix(event.currentTarget.value)} />
                <Select label="Normal source" value={splitNormalSource} onChange={setSplitNormalSource} data={datasetOptions} searchable />
                <Select label="Anomaly source" value={splitAnomalySource} onChange={setSplitAnomalySource} data={datasetOptions} searchable />
              </SimpleGrid>
              <Text size="sm" c="dimmed">Default split: normal 75% train / 12.5% validation / 12.5% holdout, anomaly 50% validation / 50% holdout.</Text>
            </Stack>
          </Paper>

          <SimpleGrid cols={{ base: 1, md: 3 }}>
            <Select label="Normal train" value={normalTrainId} onChange={setNormalTrainId} data={datasetOptions} searchable />
            <Select label="Normal validation" value={normalValidationId} onChange={setNormalValidationId} data={datasetOptions} searchable />
            <Select label="Anomaly validation" value={anomalyValidationId} onChange={setAnomalyValidationId} data={datasetOptions} searchable />
            <Select label="Normal holdout" value={normalHoldoutId} onChange={setNormalHoldoutId} data={datasetOptions} searchable clearable />
            <Select label="Anomaly holdout" value={anomalyHoldoutId} onChange={setAnomalyHoldoutId} data={datasetOptions} searchable clearable />
            <Select label="Sampler" value={sampler} onChange={(value) => setSampler((value as 'tpe' | 'random') ?? 'tpe')} data={[
              { value: 'tpe', label: 'Optuna TPE' },
              { value: 'random', label: 'Random' },
            ]} />
            <NumberInput label="Trials" value={nTrials} min={1} max={1000} onChange={(value) => setNTrials(Number(value) || 1)} />
            <NumberInput label="Parallel trials" value={maxParallelTrials} min={1} max={64} onChange={(value) => setMaxParallelTrials(Number(value) || 1)} />
          </SimpleGrid>

          <Paper withBorder p="sm" radius="sm">
            <Stack gap="sm">
              <Group justify="space-between">
                <Text fw={700}>Search space</Text>
                <Button leftSection={<Plus size={16} />} variant="subtle" onClick={() => setSearchSpace((current) => [...current, { path: '', kind: 'categorical', choices: [] }])}>
                  Add parameter
                </Button>
              </Group>
              <Table>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Path</Table.Th>
                    <Table.Th>Kind</Table.Th>
                    <Table.Th>Low</Table.Th>
                    <Table.Th>High</Table.Th>
                    <Table.Th>Choices</Table.Th>
                    <Table.Th />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {searchSpace.map((spec, index) => (
                    <Table.Tr key={`${spec.path}-${index}`}>
                      <Table.Td><TextInput value={spec.path} onChange={(event) => setSearchSpace((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, path: event.currentTarget.value } : item))} /></Table.Td>
                      <Table.Td><Select value={spec.kind} data={['categorical', 'int', 'float']} onChange={(value) => setSearchSpace((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, kind: (value as OptimizationParameterSpec['kind']) ?? 'categorical' } : item))} /></Table.Td>
                      <Table.Td><NumberInput value={spec.low ?? ''} disabled={spec.kind === 'categorical'} onChange={(value) => setSearchSpace((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, low: typeof value === 'number' ? value : null } : item))} /></Table.Td>
                      <Table.Td><NumberInput value={spec.high ?? ''} disabled={spec.kind === 'categorical'} onChange={(value) => setSearchSpace((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, high: typeof value === 'number' ? value : null } : item))} /></Table.Td>
                      <Table.Td><TextInput value={choiceText(spec)} disabled={spec.kind !== 'categorical'} onChange={(event) => setSearchSpace((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, choices: parseChoices(event.currentTarget.value) } : item))} /></Table.Td>
                      <Table.Td><Button size="xs" variant="subtle" color="red" onClick={() => setSearchSpace((current) => current.filter((_, itemIndex) => itemIndex !== index))}>Remove</Button></Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Stack>
          </Paper>

          <Group justify="flex-end">
            <Button leftSection={<Plus size={16} />} loading={pending === 'create'} onClick={handleCreateStudy}>
              Create study
            </Button>
          </Group>
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack>
          <Group justify="space-between">
            <Title order={3}>Studies</Title>
            <Group>
              <Select value={selectedStudyId} onChange={setSelectedStudyId} data={studies.map((study) => ({ value: String(study.id), label: study.name }))} placeholder="Select study" searchable />
              <Button leftSection={<Play size={16} />} loading={pending === 'start'} disabled={!selectedStudy} onClick={() => runStudyAction('start')}>Start</Button>
              <Button leftSection={<Pause size={16} />} variant="light" loading={pending === 'pause'} disabled={!selectedStudy} onClick={() => runStudyAction('pause')}>Pause</Button>
              <Button leftSection={<RotateCcw size={16} />} variant="light" loading={pending === 'resume'} disabled={!selectedStudy} onClick={() => runStudyAction('resume')}>Resume</Button>
              <Button leftSection={<Square size={16} />} variant="light" color="red" loading={pending === 'abort'} disabled={!selectedStudy} onClick={() => runStudyAction('abort')}>Abort</Button>
              <Button leftSection={<Trash2 size={16} />} variant="subtle" color="red" loading={pending === 'delete'} disabled={!selectedStudy} onClick={() => runStudyAction('delete')}>Delete</Button>
            </Group>
          </Group>

          {selectedStudy ? (
            <>
              <SimpleGrid cols={{ base: 1, md: 4 }}>
                <Paper withBorder p="sm"><Text size="xs" c="dimmed">Status</Text><Badge color={statusColor(selectedStudy.status)}>{selectedStudy.status}</Badge></Paper>
                <Paper withBorder p="sm"><Text size="xs" c="dimmed">Objective</Text><Text fw={700}>{selectedStudy.objective_name}</Text></Paper>
                <Paper withBorder p="sm"><Text size="xs" c="dimmed">Best value</Text><Text fw={700}>{selectedStudy.best_value?.toPrecision(5) ?? 'n/a'}</Text></Paper>
                <Paper withBorder p="sm"><Text size="xs" c="dimmed">Progress</Text><Text fw={700}>{selectedStudy.trials.filter((trial) => ['finished', 'failed', 'aborted'].includes(trial.status)).length}/{selectedStudy.n_trials}</Text></Paper>
              </SimpleGrid>
              {selectedStudy.error_message && <Alert color="red">{selectedStudy.error_message}</Alert>}
              <SimpleGrid cols={{ base: 1, lg: 3 }}>
                <Paper withBorder p="sm"><PlotlyChart data={objective.data} layout={objective.layout} /></Paper>
                <Paper withBorder p="sm"><PlotlyChart data={distributions.data} layout={distributions.layout} /></Paper>
                <Paper withBorder p="sm"><PlotlyChart data={parameters.data} layout={parameters.layout} /></Paper>
              </SimpleGrid>
              <Table striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Trial</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th>Phase</Table.Th>
                    <Table.Th>Objective</Table.Th>
                    <Table.Th>ROC AUC</Table.Th>
                    <Table.Th>PR AUC</Table.Th>
                    <Table.Th>Params</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {selectedStudy.trials.map((trial) => (
                    <Table.Tr key={trial.id}>
                      <Table.Td>{trial.number}</Table.Td>
                      <Table.Td><Badge color={statusColor(trial.status)}>{trial.status}</Badge></Table.Td>
                      <Table.Td>{trial.phase}</Table.Td>
                      <Table.Td>{trial.objective_value?.toPrecision(5) ?? 'n/a'}</Table.Td>
                      <Table.Td>{metricNumber(trial.metrics, 'roc_auc')?.toPrecision(4) ?? 'n/a'}</Table.Td>
                      <Table.Td>{metricNumber(trial.metrics, 'pr_auc')?.toPrecision(4) ?? 'n/a'}</Table.Td>
                      <Table.Td><Text size="xs" className="path-text">{JSON.stringify(trial.sampled_params)}</Text></Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </>
          ) : (
            <Alert color="blue">Create or select an optimization study.</Alert>
          )}
        </Stack>
      </Paper>
    </Stack>
  );
}
