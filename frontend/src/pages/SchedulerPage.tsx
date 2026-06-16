import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  NumberInput,
  Paper,
  Progress,
  ScrollArea,
  Select,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';

import { StepCard } from '../components/StepCard';
import { FileText, Info, RotateCcw, Search, StopCircle, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  abortTestingRun,
  abortTrainingRun,
  clearHeatmaps,
  deleteTestingRun,
  deleteTrainingRun,
  getSchedulerSettings,
  getTestingRunLog,
  getTrainingRunLog,
  listMethodConfigurations,
  listMethodDefinitions,
  listPreprocessingPipelines,
  listHeatmaps,
  listTestingRuns,
  listTrainingPipelines,
  listTrainingRuns,
  restartTestingRun,
  restartTrainingRun,
  updateSchedulerSettings,
} from '../api';
import { SchedulerDetailsModal } from '../training/SchedulerDetailsModal';
import type { SchedulerJob } from '../training/SchedulerDetailsModal';
import { formatDuration, runStatusColor } from '../training/runStatus';
import type {
  MethodConfiguration,
  MethodDefinition,
  HeatmapRun,
  PreprocessingPipeline,
  SchedulerSettings,
  TestingRun,
  TrainingPipeline,
  TrainingRun,
} from '../types';

const TERMINAL = new Set(['finished', 'failed', 'aborted']);

type HeatmapGroup = {
  key: string;
  type: 'single' | 'video';
  testingRunId: number;
  count: number;
  createdAt: string;
  startTimestamp: string;
  endTimestamp: string;
  status: string;
  maxHeatmap: HeatmapRun;
  meanError: number;
};

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function jobKey(job: SchedulerJob): string {
  return `${job.kind}-${job.run.id}`;
}

function jobName(job: SchedulerJob): string {
  return job.kind === 'train' ? job.run.training_pipeline_name : job.run.name;
}

function summarizeHeatmaps(heatmaps: HeatmapRun[]): HeatmapGroup[] {
  const byRun = new Map<number, HeatmapRun[]>();
  for (const heatmap of heatmaps) {
    const entries = byRun.get(heatmap.testing_run_id) ?? [];
    entries.push(heatmap);
    byRun.set(heatmap.testing_run_id, entries);
  }
  return [...byRun.entries()]
    .map(([testingRunId, entries]) => {
      const sorted = [...entries].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
      const maxHeatmap = sorted.reduce((best, current) => (current.max_error > best.max_error ? current : best), sorted[0]);
      const status = sorted.some((entry) => entry.status === 'running')
        ? 'running'
        : sorted.some((entry) => entry.status === 'failed')
          ? 'failed'
          : 'finished';
      return {
        key: String(testingRunId),
        type: (sorted.length > 1 ? 'video' : 'single') as HeatmapGroup['type'],
        testingRunId,
        count: sorted.length,
        createdAt: sorted.map((entry) => entry.created_at).sort().at(-1) ?? sorted[0].created_at,
        startTimestamp: sorted[0].timestamp,
        endTimestamp: sorted[sorted.length - 1].timestamp,
        status,
        maxHeatmap,
        meanError: sorted.reduce((sum, entry) => sum + entry.mean_error, 0) / sorted.length,
      };
    })
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

function ProgressCell({ job }: { job: SchedulerJob }) {
  if (job.kind === 'test') {
    return <Text size="sm">{job.run.image_count != null ? `${job.run.image_count} imgs` : '—'}</Text>;
  }
  const run = job.run;
  if (run.builder_kind === 'form') {
    return <Text size="sm">{run.image_count != null ? `${run.image_count} imgs` : '—'}</Text>;
  }
  const total = run.epochs_total ?? run.epochs ?? 0;
  const value = total > 0 ? (run.epochs_completed / total) * 100 : 0;
  return (
    <Stack gap={2}>
      <Text size="xs">{run.epochs_completed}/{total || '?'} epochs</Text>
      {total > 0 && <Progress value={value} size="sm" radius="sm" color={runStatusColor(run.status)} />}
    </Stack>
  );
}

function LogModal({ job, onClose }: { job: SchedulerJob | null; onClose: () => void }) {
  const [log, setLog] = useState('');
  useEffect(() => {
    if (!job) return undefined;
    let cancelled = false;
    const load = () => {
      const fetcher = job.kind === 'train' ? getTrainingRunLog : getTestingRunLog;
      fetcher(job.run.id)
        .then((result) => {
          if (!cancelled) setLog(result.log);
        })
        .catch(() => undefined);
    };
    load();
    const active = job.run.status === 'running' || job.run.status === 'queued';
    const interval = active ? window.setInterval(load, 2000) : null;
    return () => {
      cancelled = true;
      if (interval) window.clearInterval(interval);
    };
  }, [job]);

  return (
    <Modal opened={job !== null} onClose={onClose} title={job ? jobName(job) : ''} size="xl">
      <Paper withBorder p="xs" radius="sm">
        <ScrollArea h={420}>
          <Text size="xs" ff="monospace" style={{ whiteSpace: 'pre-wrap' }}>
            {log || 'No log output yet.'}
          </Text>
        </ScrollArea>
      </Paper>
    </Modal>
  );
}

export function SchedulerPage({ active = true }: { active?: boolean }) {
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [testingRuns, setTestingRuns] = useState<TestingRun[]>([]);
  const [pipelines, setPipelines] = useState<TrainingPipeline[]>([]);
  const [preprocessing, setPreprocessing] = useState<PreprocessingPipeline[]>([]);
  const [methods, setMethods] = useState<MethodConfiguration[]>([]);
  const [methodDefs, setMethodDefs] = useState<MethodDefinition[]>([]);
  const [heatmaps, setHeatmaps] = useState<HeatmapRun[]>([]);
  const [schedulerSettings, setSchedulerSettings] = useState<SchedulerSettings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<{ max_gpu_slots: number; only_gpu: boolean } | null>(null);
  const [savingSettings, setSavingSettings] = useState(false);

  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [methodFilter, setMethodFilter] = useState<string | null>(null);

  const [logJob, setLogJob] = useState<SchedulerJob | null>(null);
  const [detailJob, setDetailJob] = useState<SchedulerJob | null>(null);

  async function refreshRuns() {
    const [nextTraining, nextTesting, nextHeatmaps] = await Promise.all([listTrainingRuns(), listTestingRuns(), listHeatmaps()]);
    setTrainingRuns(nextTraining);
    setTestingRuns(nextTesting);
    setHeatmaps(nextHeatmaps);
  }

  async function refreshSchedulerSettings() {
    const next = await getSchedulerSettings();
    setSchedulerSettings(next);
    setSettingsDraft({ max_gpu_slots: next.max_gpu_slots, only_gpu: next.only_gpu });
  }

  async function refreshReferences() {
    const [nextPipelines, nextPre, nextMethods, nextDefs] = await Promise.all([
      listTrainingPipelines(),
      listPreprocessingPipelines(),
      listMethodConfigurations(),
      listMethodDefinitions(),
    ]);
    setPipelines(nextPipelines);
    setPreprocessing(nextPre);
    setMethods(nextMethods);
    setMethodDefs(nextDefs);
  }

  useEffect(() => {
    if (!active) return undefined;
    refreshReferences().catch(() => undefined);
    refreshSchedulerSettings().catch(() => undefined);
    refreshRuns().catch((error) => notifyError('Could not load scheduler', error));
    const interval = window.setInterval(() => {
      refreshRuns().catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const pipelineById = useMemo(() => new Map(pipelines.map((p) => [p.id, p])), [pipelines]);
  const preprocessingById = useMemo(() => new Map(preprocessing.map((p) => [p.id, p])), [preprocessing]);
  const methodById = useMemo(() => new Map(methods.map((m) => [m.id, m])), [methods]);
  const methodByType = useMemo(() => new Map(methodDefs.map((d) => [d.type, d])), [methodDefs]);
  const trainingRunById = useMemo(() => new Map(trainingRuns.map((r) => [r.id, r])), [trainingRuns]);

  const jobs = useMemo<SchedulerJob[]>(() => {
    const list: SchedulerJob[] = [
      ...trainingRuns.map((run) => ({ kind: 'train' as const, run })),
      ...testingRuns.map((run) => ({ kind: 'test' as const, run })),
    ];
    return list.sort((a, b) => (b.run.created_at ?? '').localeCompare(a.run.created_at ?? ''));
  }, [trainingRuns, testingRuns]);

  const methodOptions = useMemo(
    () => [...new Set(jobs.map((job) => job.run.method_type))].map((type) => ({ value: type, label: type })),
    [jobs],
  );

  const filteredJobs = useMemo(() => {
    const query = search.trim().toLowerCase();
    return jobs.filter((job) => {
      if (typeFilter && job.kind !== typeFilter) return false;
      if (statusFilter && job.run.status !== statusFilter) return false;
      if (methodFilter && job.run.method_type !== methodFilter) return false;
      if (!query) return true;
      return jobName(job).toLowerCase().includes(query) || job.run.method_type.toLowerCase().includes(query);
    });
  }, [jobs, search, typeFilter, statusFilter, methodFilter]);

  const heatmapGroups = useMemo(() => summarizeHeatmaps(heatmaps), [heatmaps]);

  async function withRefresh(action: () => Promise<unknown>, errorTitle: string) {
    try {
      await action();
      await refreshRuns();
    } catch (error) {
      notifyError(errorTitle, error);
    }
  }

  function handleAbort(job: SchedulerJob) {
    const action = job.kind === 'train' ? () => abortTrainingRun(job.run.id) : () => abortTestingRun(job.run.id);
    withRefresh(action, 'Could not abort');
  }

  function handleRestart(job: SchedulerJob) {
    const action = job.kind === 'train' ? () => restartTrainingRun(job.run.id) : () => restartTestingRun(job.run.id);
    withRefresh(action, 'Could not restart');
  }

  function handleDelete(job: SchedulerJob) {
    if (!window.confirm(`Remove ${job.kind === 'train' ? 'training run' : 'inference'} "${jobName(job)}"?`)) return;
    const action = job.kind === 'train' ? () => deleteTrainingRun(job.run.id) : () => deleteTestingRun(job.run.id);
    withRefresh(action, 'Could not remove');
  }

  async function handleSaveSettings() {
    if (!settingsDraft) return;
    setSavingSettings(true);
    try {
      const next = await updateSchedulerSettings(settingsDraft);
      setSchedulerSettings(next);
      setSettingsDraft({ max_gpu_slots: next.max_gpu_slots, only_gpu: next.only_gpu });
      notifications.show({ color: 'green', title: 'Scheduler settings saved', message: 'GPU queue settings were updated.' });
    } catch (error) {
      notifyError('Could not save scheduler settings', error);
    } finally {
      setSavingSettings(false);
    }
  }

  async function handleClearHeatmaps() {
    if (!window.confirm('Clear all cached heatmaps?')) return;
    try {
      await clearHeatmaps();
      setHeatmaps([]);
    } catch (error) {
      notifyError('Could not clear heatmaps', error);
    }
  }

  return (
    <Stack gap="lg">
      <div>
        <Title order={2}>Scheduler</Title>
        <Text c="dimmed" size="sm">
          All training and inference jobs across the GPU queue — queued, running, finished, failed, aborted.
        </Text>
      </div>

      <Paper withBorder p="md" radius="sm" style={{ borderLeft: '4px solid var(--mantine-color-blue-5)' }}>
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <div>
              <Title order={3}>GPU queue settings</Title>
              <Text size="sm" c="dimmed">
                Applies to training and scheduled inference jobs. CPU heatmap calculations run separately.
              </Text>
            </div>
            <Badge variant="light" color={schedulerSettings?.detected_gpu_count ? 'grape' : 'gray'}>
              {schedulerSettings ? `${schedulerSettings.detected_gpu_count} GPUs detected` : 'Detecting GPUs'}
            </Badge>
          </Group>
          <Group align="end">
            <NumberInput
              label="GPU slots"
              description="Maximum selected GPU workers used by the queue."
              min={1}
              max={Math.max(1, schedulerSettings?.detected_gpu_count || settingsDraft?.max_gpu_slots || 1)}
              value={settingsDraft?.max_gpu_slots ?? 1}
              onChange={(value) =>
                setSettingsDraft((current) => ({
                  max_gpu_slots: typeof value === 'number' && Number.isFinite(value) ? value : current?.max_gpu_slots ?? 1,
                  only_gpu: current?.only_gpu ?? false,
                }))
              }
            />
            <Switch
              label="Only run scheduled jobs when a GPU slot is available"
              checked={settingsDraft?.only_gpu ?? false}
              onChange={(event) =>
                setSettingsDraft((current) => ({
                  max_gpu_slots: current?.max_gpu_slots ?? 1,
                  only_gpu: event.currentTarget.checked,
                }))
              }
            />
            <Button onClick={handleSaveSettings} loading={savingSettings} disabled={!settingsDraft}>
              Save settings
            </Button>
          </Group>
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group grow>
            <TextInput
              placeholder="Search by name or method"
              leftSection={<Search size={16} />}
              value={search}
              onChange={(event) => setSearch(event.currentTarget.value)}
            />
            <Select
              placeholder="Type"
              data={[
                { value: 'train', label: 'Training' },
                { value: 'test', label: 'Inference' },
              ]}
              value={typeFilter}
              onChange={setTypeFilter}
              clearable
            />
            <Select
              placeholder="Status"
              data={['queued', 'running', 'finished', 'failed', 'aborted']}
              value={statusFilter}
              onChange={setStatusFilter}
              clearable
            />
            <Select placeholder="Method type" data={methodOptions} value={methodFilter} onChange={setMethodFilter} clearable />
          </Group>
        </Stack>
      </Paper>

      <StepCard title="Jobs" color="cyan">
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm" miw={1100}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Type</Table.Th>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>Method</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Device</Table.Th>
                  <Table.Th>Progress</Table.Th>
                  <Table.Th>Duration</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {filteredJobs.map((job) => {
                  const run = job.run;
                  const terminal = TERMINAL.has(run.status);
                  const abortable = run.status === 'queued' || run.status === 'running';
                  return (
                    <Table.Tr key={jobKey(job)}>
                      <Table.Td>
                        <Badge color={job.kind === 'train' ? 'blue' : 'grape'} variant="light">
                          {job.kind === 'train' ? 'Training' : 'Inference'}
                        </Badge>
                      </Table.Td>
                      <Table.Td>{jobName(job)}</Table.Td>
                      <Table.Td>
                        <Text size="sm">{run.method_type}</Text>
                      </Table.Td>
                      <Table.Td>
                        <Badge color={runStatusColor(run.status)}>{run.status}</Badge>
                      </Table.Td>
                      <Table.Td>
                        {run.device ? (
                          <Badge variant="light" color={run.device === 'CPU' ? 'gray' : 'grape'}>{run.device}</Badge>
                        ) : (
                          '—'
                        )}
                      </Table.Td>
                      <Table.Td><ProgressCell job={job} /></Table.Td>
                      <Table.Td>{formatDuration(run.duration_seconds)}</Table.Td>
                      <Table.Td>
                        <Group gap="xs" justify="flex-end" wrap="nowrap">
                          <Tooltip label="Details">
                            <ActionIcon variant="subtle" onClick={() => setDetailJob(job)}>
                              <Info size={18} />
                            </ActionIcon>
                          </Tooltip>
                          <Tooltip label="Logs">
                            <ActionIcon variant="subtle" onClick={() => setLogJob(job)}>
                              <FileText size={18} />
                            </ActionIcon>
                          </Tooltip>
                          {abortable && (
                            <Tooltip label="Abort">
                              <ActionIcon color="orange" variant="subtle" onClick={() => handleAbort(job)}>
                                <StopCircle size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                          {terminal && (
                            <Tooltip label="Restart">
                              <ActionIcon variant="subtle" onClick={() => handleRestart(job)}>
                                <RotateCcw size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                          {run.status !== 'running' && (
                            <Tooltip label="Remove">
                              <ActionIcon color="red" variant="subtle" onClick={() => handleDelete(job)}>
                                <Trash2 size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  );
                })}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {filteredJobs.length === 0 && <Alert color="blue">No jobs match the current filters.</Alert>}
      </StepCard>

      <Paper withBorder p="md" radius="sm" style={{ borderLeft: '4px solid var(--mantine-color-grape-5)' }}>
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <div>
              <Title order={3}>Heatmaps</Title>
              <Text size="sm" c="dimmed">
                Cached CPU pixel-error heatmaps computed from Analysis. These do not occupy GPU queue slots.
              </Text>
            </div>
            <Button variant="default" color="red" leftSection={<Trash2 size={16} />} disabled={heatmaps.length === 0} onClick={handleClearHeatmaps}>
              Clear heatmaps
            </Button>
          </Group>
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm" miw={980}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Created</Table.Th>
                  <Table.Th>Type</Table.Th>
                  <Table.Th>Testing run</Table.Th>
                  <Table.Th>Frames</Table.Th>
                  <Table.Th>Time range</Table.Th>
                  <Table.Th>Max pixel</Table.Th>
                  <Table.Th>Max error</Table.Th>
                  <Table.Th>Mean error</Table.Th>
                  <Table.Th>Status</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {heatmapGroups.map((group) => (
                  <Table.Tr key={group.key}>
                    <Table.Td>{new Date(group.createdAt).toLocaleString()}</Table.Td>
                    <Table.Td>
                      <Badge variant="light" color={group.type === 'video' ? 'red' : 'blue'}>
                        {group.type === 'video' ? 'Heatmap video' : 'Single heatmap'}
                      </Badge>
                    </Table.Td>
                    <Table.Td>#{group.testingRunId}</Table.Td>
                    <Table.Td>{group.count}</Table.Td>
                    <Table.Td>
                      {group.type === 'single'
                        ? new Date(group.startTimestamp).toLocaleString()
                        : `${new Date(group.startTimestamp).toLocaleString()} → ${new Date(group.endTimestamp).toLocaleString()}`}
                    </Table.Td>
                    <Table.Td>
                      ({group.maxHeatmap.max_x}, {group.maxHeatmap.max_y})
                    </Table.Td>
                    <Table.Td>{group.maxHeatmap.max_error.toExponential(3)}</Table.Td>
                    <Table.Td>{group.meanError.toExponential(3)}</Table.Td>
                    <Table.Td>
                      <Badge color={group.status === 'finished' ? 'green' : group.status === 'running' ? 'blue' : 'red'}>{group.status}</Badge>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {heatmapGroups.length === 0 && <Alert color="blue">Computed heatmaps will appear here.</Alert>}
        </Stack>
      </Paper>

      <LogModal job={logJob} onClose={() => setLogJob(null)} />
      <SchedulerDetailsModal
        job={detailJob}
        onClose={() => setDetailJob(null)}
        pipelineById={pipelineById}
        preprocessingById={preprocessingById}
        methodById={methodById}
        methodByType={methodByType}
        trainingRunById={trainingRunById}
      />
    </Stack>
  );
}
