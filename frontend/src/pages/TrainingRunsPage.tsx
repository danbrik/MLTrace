import { Group, Paper, Select, Stack, Text, TextInput, Title } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Search } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  abortTrainingRun,
  deleteTrainingRun,
  enqueueTrainingRun,
  listMethodConfigurations,
  listPreprocessingPipelines,
  listTrainingPipelines,
  listTrainingRuns,
  restartTrainingRun,
} from '../api';
import { RunLogModal } from '../training/RunLogModal';
import { StartRunPanel } from '../training/StartRunPanel';
import { TrainingRunsTable } from '../training/TrainingRunsTable';
import type { MethodConfiguration, PreprocessingPipeline, TrainingPipeline, TrainingRun } from '../types';

const STATUS_OPTIONS = ['queued', 'running', 'finished', 'failed', 'aborted'];
const ACTIVE_STATUSES = new Set(['queued', 'running']);

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

export function TrainingRunsPage({ active = true }: { active?: boolean }) {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [pipelines, setPipelines] = useState<TrainingPipeline[]>([]);
  const [preprocessing, setPreprocessing] = useState<PreprocessingPipeline[]>([]);
  const [methods, setMethods] = useState<MethodConfiguration[]>([]);
  const [enqueuing, setEnqueuing] = useState(false);
  const [logRun, setLogRun] = useState<TrainingRun | null>(null);

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [methodFilter, setMethodFilter] = useState<string | null>(null);
  const [trainingModeFilter, setTrainingModeFilter] = useState<string | null>(null);

  // Keep the latest filter values in a ref so the poll interval always reads
  // fresh values without re-creating the timer on every keystroke.
  const filtersRef = useRef({ search, statusFilter, methodFilter, trainingModeFilter });
  filtersRef.current = { search, statusFilter, methodFilter, trainingModeFilter };

  async function refresh() {
    const current = filtersRef.current;
    const [nextRuns, nextPipelines, nextPreprocessing, nextMethods] = await Promise.all([
      listTrainingRuns({
        search: current.search || undefined,
        status: current.statusFilter,
        method_type: current.methodFilter,
        training_mode: current.trainingModeFilter,
      }),
      listTrainingPipelines(),
      listPreprocessingPipelines(),
      listMethodConfigurations(),
    ]);
    setRuns(nextRuns);
    setPipelines(nextPipelines);
    setPreprocessing(nextPreprocessing);
    setMethods(nextMethods);
    setLogRun((open) => (open ? nextRuns.find((run) => run.id === open.id) ?? null : null));
  }

  // Poll every 2s while the page is active so statuses update live.
  useEffect(() => {
    if (!active) return undefined;
    refresh().catch((error) => notifyError('Could not load training runs', error));
    const interval = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const busyPipelineIds = useMemo(
    () => new Set(runs.filter((run) => ACTIVE_STATUSES.has(run.status)).map((run) => run.training_pipeline_id)),
    [runs],
  );
  const methodOptions = useMemo(
    () => [...new Set(runs.map((run) => run.method_type))].map((type) => ({ value: type, label: type })),
    [runs],
  );

  async function handleEnqueue(pipelineId: number) {
    setEnqueuing(true);
    try {
      const run = await enqueueTrainingRun(pipelineId);
      await refresh();
      notifications.show({ color: 'green', title: 'Training queued', message: run.training_pipeline_name });
    } catch (error) {
      notifyError('Could not enqueue training', error);
    } finally {
      setEnqueuing(false);
    }
  }

  async function handleAbort(run: TrainingRun) {
    try {
      await abortTrainingRun(run.id);
      await refresh();
      notifications.show({ color: 'orange', title: 'Abort requested', message: run.training_pipeline_name });
    } catch (error) {
      notifyError('Could not abort run', error);
    }
  }

  async function handleRestart(run: TrainingRun) {
    try {
      await restartTrainingRun(run.id);
      await refresh();
      notifications.show({ color: 'green', title: 'Run re-queued', message: run.training_pipeline_name });
    } catch (error) {
      notifyError('Could not restart run', error);
    }
  }

  async function handleDelete(run: TrainingRun) {
    if (!window.confirm(`Remove the run for "${run.training_pipeline_name}"?`)) return;
    try {
      await deleteTrainingRun(run.id);
      await refresh();
      notifications.show({ color: 'green', title: 'Run removed', message: run.training_pipeline_name });
    } catch (error) {
      notifyError('Could not remove run', error);
    }
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Training Runs</Title>
        <Text c="dimmed" size="sm">
          Queue training pipelines, run them in parallel across GPUs, and track losses, duration, and artifacts.
        </Text>
      </div>

      <StartRunPanel
        pipelines={pipelines}
        preprocessingPipelines={preprocessing}
        methodConfigurations={methods}
        busyPipelineIds={busyPipelineIds}
        onEnqueue={handleEnqueue}
        loading={enqueuing}
      />

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Title order={3}>Filters</Title>
          <Group grow>
            <TextInput
              placeholder="Search by pipeline or dataset name"
              leftSection={<Search size={16} />}
              value={search}
              onChange={(event) => setSearch(event.currentTarget.value)}
            />
            <Select placeholder="Status" data={STATUS_OPTIONS} value={statusFilter} onChange={setStatusFilter} clearable />
            <Select placeholder="Method type" data={methodOptions} value={methodFilter} onChange={setMethodFilter} clearable />
            <Select
              placeholder="Training mode"
              data={[
                { value: 'gradient', label: 'Gradient training' },
                { value: 'fit', label: 'Training' },
                { value: 'none', label: 'No Training' },
              ]}
              value={trainingModeFilter}
              onChange={setTrainingModeFilter}
              clearable
            />
          </Group>
        </Stack>
      </Paper>

      <TrainingRunsTable
        runs={runs}
        onAbort={handleAbort}
        onRestart={handleRestart}
        onDelete={handleDelete}
        onShowLog={setLogRun}
      />

      <RunLogModal run={logRun} onClose={() => setLogRun(null)} />
    </Stack>
  );
}
