import {
  ActionIcon,
  Accordion,
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  NumberInput,
  Paper,
  Progress,
  ScrollArea,
  SegmentedControl,
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
import { usePendingIds } from '../hooks/usePendingIds';
import { ArrowDown, ArrowUp, FileText, Info, RotateCcw, Search, StopCircle, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  abortHeatmapRange,
  abortTestingRun,
  abortTrainingRun,
  clearHeatmaps,
  deleteHeatmapRange,
  deleteTestingRun,
  deleteTrainingRun,
  getHeatmapRangeLog,
  getSchedulerSettings,
  getGpuUsage,
  getTestingRunLog,
  getTrainingRunLog,
  listHeatmapRanges,
  listMethodConfigurations,
  listMethodDefinitions,
  listPreprocessingPipelines,
  listHeatmaps,
  listTestingRuns,
  listTrainingPipelines,
  listTrainingRuns,
  listSchedulerJobs,
  moveSchedulerJob,
  restartTestingRun,
  restartTrainingRun,
  updateSchedulerSettings,
} from '../api';
import { SchedulerDetailsModal } from '../training/SchedulerDetailsModal';
import type { SchedulerJob } from '../training/SchedulerDetailsModal';
import { formatDuration, runStatusColor } from '../training/runStatus';
import type {
  HeatmapRangeRun,
  MethodConfiguration,
  MethodDefinition,
  HeatmapRunSummary,
  PreprocessingPipeline,
  SchedulerSettings,
  TestingRun,
  TrainingPipeline,
  TrainingRun,
  GpuSnapshot,
  SchedulerJobWithProject,
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
  maxHeatmap: HeatmapRunSummary;
  meanError: number;
};

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

type DisplayJob = SchedulerJob & { project_id?: string; project_name?: string };

function jobKey(job: DisplayJob): string {
  return `${job.project_id ?? 'current'}-${job.kind}-${job.run.id}`;
}

function jobName(job: SchedulerJob): string {
  if (job.kind === 'train') return job.run.training_pipeline_name;
  if (job.kind === 'heatmap') return `Heatmap video · ${job.run.testing_run_name}`;
  return job.run.name;
}

function jobMethodType(job: SchedulerJob): string {
  return job.kind === 'heatmap' ? 'heatmap video' : job.run.method_type;
}

function summarizeHeatmaps(heatmaps: HeatmapRunSummary[]): HeatmapGroup[] {
  const byRun = new Map<number, HeatmapRunSummary[]>();
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
  if (job.kind === 'heatmap') {
    const done = job.run.done_count ?? 0;
    const total = job.run.frame_count ?? null;
    if (total && total > 0) {
      return (
        <Stack gap={2}>
          <Text size="xs">{done}/{total} frames</Text>
          <Progress value={Math.min(100, (done / total) * 100)} size="sm" radius="sm" color={runStatusColor(job.run.status)} />
        </Stack>
      );
    }
    return <Text size="xs" c="dimmed">{done > 0 ? `${done} frames` : '—'}</Text>;
  }
  if (job.kind === 'test') {
    const processed = job.run.image_count ?? 0;
    const expected = job.run.expected_image_count ?? (job.run.status === 'finished' ? job.run.image_count : null);
    if (expected && expected > 0) {
      return (
        <Stack gap={2}>
          <Text size="xs">
            {processed}/{expected} images
          </Text>
          <Progress value={Math.min(100, (processed / expected) * 100)} size="sm" radius="sm" color={runStatusColor(job.run.status)} />
        </Stack>
      );
    }
    return <Text size="sm">{job.run.image_count != null ? `${job.run.image_count}/? images` : '—'}</Text>;
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

function LogModal({ job, onClose }: { job: DisplayJob | null; onClose: () => void }) {
  const [log, setLog] = useState('');
  useEffect(() => {
    if (!job) return undefined;
    let cancelled = false;
    const load = () => {
      const fetcher =
        job.kind === 'train' ? getTrainingRunLog : job.kind === 'heatmap' ? getHeatmapRangeLog : getTestingRunLog;
      fetcher(job.run.id, job.project_id)
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
  const [heatmaps, setHeatmaps] = useState<HeatmapRunSummary[]>([]);
  const [heatmapRanges, setHeatmapRanges] = useState<HeatmapRangeRun[]>([]);
  const [schedulerSettings, setSchedulerSettings] = useState<SchedulerSettings | null>(null);
  const [gpuUsage, setGpuUsage] = useState<GpuSnapshot | null>(null);
  const [scope, setScope] = useState<'project' | 'all'>('project');
  const [globalJobs, setGlobalJobs] = useState<SchedulerJobWithProject[]>([]);
  const [settingsDraft, setSettingsDraft] = useState<{ max_gpu_slots: number; only_gpu: boolean } | null>(null);
  const [savingSettings, setSavingSettings] = useState(false);
  const [clearingHeatmaps, setClearingHeatmaps] = useState(false);
  const rowActions = usePendingIds();

  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [methodFilter, setMethodFilter] = useState<string | null>(null);

  const [logJob, setLogJob] = useState<DisplayJob | null>(null);
  const [detailJob, setDetailJob] = useState<SchedulerJob | null>(null);

  async function refreshRuns() {
    if (scope === 'all') {
      setGlobalJobs(await listSchedulerJobs('all'));
      return;
    }
    const [nextTraining, nextTesting, nextHeatmaps, nextHeatmapRanges] = await Promise.all([
      listTrainingRuns(),
      listTestingRuns(),
      listHeatmaps(),
      listHeatmapRanges(),
    ]);
    setTrainingRuns(nextTraining);
    setTestingRuns(nextTesting);
    setHeatmaps(nextHeatmaps);
    setHeatmapRanges(nextHeatmapRanges);
  }

  async function refreshGpu(force = false) {
    setGpuUsage(await getGpuUsage(force));
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
    refreshGpu(true).catch(() => undefined);
    refreshRuns().catch((error) => notifyError('Could not load scheduler', error));
    const interval = window.setInterval(() => {
      refreshRuns().catch(() => undefined);
    }, 2000);
    const gpuInterval = window.setInterval(() => refreshGpu().catch(() => undefined), 60_000);
    return () => {
      window.clearInterval(interval);
      window.clearInterval(gpuInterval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, scope]);

  const pipelineById = useMemo(() => new Map(pipelines.map((p) => [p.id, p])), [pipelines]);
  const preprocessingById = useMemo(() => new Map(preprocessing.map((p) => [p.id, p])), [preprocessing]);
  const methodById = useMemo(() => new Map(methods.map((m) => [m.id, m])), [methods]);
  const methodByType = useMemo(() => new Map(methodDefs.map((d) => [d.type, d])), [methodDefs]);
  const trainingRunById = useMemo(() => new Map(trainingRuns.map((r) => [r.id, r])), [trainingRuns]);

  const jobs = useMemo<DisplayJob[]>(() => {
    if (scope === 'all') {
      return globalJobs.map((job) => ({
        kind: job.kind,
        run: job.run,
        project_id: job.project_id,
        project_name: job.project_name,
      })) as DisplayJob[];
    }
    const list: DisplayJob[] = [
      ...trainingRuns.map((run) => ({ kind: 'train' as const, run })),
      ...testingRuns.map((run) => ({ kind: 'test' as const, run })),
      ...heatmapRanges.map((run) => ({ kind: 'heatmap' as const, run })),
    ];
    return list.sort((a, b) => {
      const aQueued = a.run.status === 'queued';
      const bQueued = b.run.status === 'queued';
      if (aQueued && bQueued) {
        const aRank = a.run.queue_rank ?? Number.MAX_SAFE_INTEGER;
        const bRank = b.run.queue_rank ?? Number.MAX_SAFE_INTEGER;
        if (aRank !== bRank) return aRank - bRank;
        return (a.run.enqueued_at ?? '').localeCompare(b.run.enqueued_at ?? '') || a.run.id - b.run.id;
      }
      if (aQueued !== bQueued) return aQueued ? -1 : 1;
      return (b.run.created_at ?? '').localeCompare(a.run.created_at ?? '');
    });
  }, [trainingRuns, testingRuns, heatmapRanges, globalJobs, scope]);

  const queuedJobs = useMemo(() => jobs.filter((job) => job.run.status === 'queued'), [jobs]);
  const queueIndexByKey = useMemo(
    () => new Map(queuedJobs.map((job, index) => [jobKey(job), index])),
    [queuedJobs],
  );

  const methodOptions = useMemo(
    () => [...new Set(jobs.map((job) => jobMethodType(job)))].map((type) => ({ value: type, label: type })),
    [jobs],
  );

  const filteredJobs = useMemo(() => {
    const query = search.trim().toLowerCase();
    return jobs.filter((job) => {
      if (typeFilter && job.kind !== typeFilter) return false;
      if (statusFilter && job.run.status !== statusFilter) return false;
      if (methodFilter && jobMethodType(job) !== methodFilter) return false;
      if (!query) return true;
      return jobName(job).toLowerCase().includes(query) || jobMethodType(job).toLowerCase().includes(query);
    });
  }, [jobs, search, typeFilter, statusFilter, methodFilter]);

  const heatmapGroups = useMemo(() => summarizeHeatmaps(heatmaps), [heatmaps]);

  async function withRefresh(actionId: string, action: () => Promise<unknown>, errorTitle: string) {
    await rowActions.runPending(actionId, async () => {
      await action();
      await refreshRuns();
    }).catch((error) => {
      notifyError(errorTitle, error);
    });
  }

  function handleAbort(job: DisplayJob) {
    const action =
      job.kind === 'train'
        ? () => abortTrainingRun(job.run.id, job.project_id)
        : job.kind === 'heatmap'
          ? () => abortHeatmapRange(job.run.id, job.project_id)
          : () => abortTestingRun(job.run.id, job.project_id);
    withRefresh(`abort:${jobKey(job)}`, action, 'Could not abort');
  }

  function handleRestart(job: DisplayJob) {
    if (job.kind === 'heatmap') return; // heatmap videos are not restartable; re-render from Analysis
    const action = job.kind === 'train'
      ? () => restartTrainingRun(job.run.id, job.project_id)
      : () => restartTestingRun(job.run.id, job.project_id);
    withRefresh(`restart:${jobKey(job)}`, action, 'Could not restart');
  }

  function handleDelete(job: DisplayJob) {
    const label = job.kind === 'train' ? 'training run' : job.kind === 'heatmap' ? 'heatmap video' : 'inference';
    if (!window.confirm(`Remove ${label} "${jobName(job)}"?`)) return;
    const action =
      job.kind === 'train'
        ? () => deleteTrainingRun(job.run.id, job.project_id)
        : job.kind === 'heatmap'
          ? () => deleteHeatmapRange(job.run.id, job.project_id)
          : () => deleteTestingRun(job.run.id, job.project_id);
    withRefresh(`delete:${jobKey(job)}`, action, 'Could not remove');
  }

  function handleMove(job: DisplayJob, direction: 'up' | 'down') {
    withRefresh(`move-${direction}:${jobKey(job)}`, () => moveSchedulerJob(job.kind, job.run.id, direction, job.project_id), 'Could not move job');
  }

  async function handleSaveSettings() {
    if (!settingsDraft) return;
    if (savingSettings) return;
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
    if (clearingHeatmaps) return;
    setClearingHeatmaps(true);
    try {
      await clearHeatmaps();
      setHeatmaps([]);
    } catch (error) {
      notifyError('Could not clear heatmaps', error);
    } finally {
      setClearingHeatmaps(false);
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

      <Accordion variant="contained">
        <Accordion.Item value="gpu-live">
          <Accordion.Control>
            <Group justify="space-between" pr="md">
              <div>
                <Text fw={700}>Live GPU usage</Text>
                <Text size="xs" c="dimmed">
                  {gpuUsage
                    ? `MLTrace uses ${gpuUsage.mltrace_memory_mb} MiB across ${gpuUsage.gpu_slots} GPU slots · ${gpuUsage.running_jobs} running · ${gpuUsage.queued_jobs} queued`
                    : 'Loading NVIDIA and MLTrace usage…'}
                </Text>
              </div>
              {gpuUsage && <Badge color={gpuUsage.available ? 'green' : 'gray'}>{gpuUsage.available ? 'nvidia-smi live' : 'unavailable'}</Badge>}
            </Group>
          </Accordion.Control>
          <Accordion.Panel>
            <Stack gap="md">
              {gpuUsage?.error && <Alert color="yellow">{gpuUsage.error}</Alert>}
              {gpuUsage && (
                <Text size="xs" c="dimmed">Snapshot {new Date(gpuUsage.captured_at).toLocaleString()} · refreshed every minute</Text>
              )}
              {gpuUsage?.devices.map((device) => (
                <Paper key={device.uuid} withBorder p="sm">
                  <Group justify="space-between">
                    <Text fw={600}>GPU {device.index} · {device.name}</Text>
                    <Group gap="xs">
                      <Badge variant="light">{device.utilization_percent}% utilization</Badge>
                      <Badge variant="light" color="grape">{device.memory_used_mb}/{device.memory_total_mb} MiB</Badge>
                      <Badge variant="light" color="blue">MLTrace {device.mltrace_memory_mb} MiB</Badge>
                      {device.temperature_c != null && <Badge variant="light" color="orange">{device.temperature_c} °C</Badge>}
                    </Group>
                  </Group>
                </Paper>
              ))}
              {gpuUsage && gpuUsage.projects.length > 0 && (
                <Table striped withTableBorder>
                  <Table.Thead><Table.Tr><Table.Th>Project</Table.Th><Table.Th>GPU memory</Table.Th><Table.Th>Slots</Table.Th><Table.Th>Running</Table.Th><Table.Th>Queued</Table.Th></Table.Tr></Table.Thead>
                  <Table.Tbody>{gpuUsage.projects.map((usage) => (
                    <Table.Tr key={usage.project_id}><Table.Td>{usage.project_name}</Table.Td><Table.Td>{usage.gpu_memory_mb} MiB</Table.Td><Table.Td>{usage.gpu_slots}</Table.Td><Table.Td>{usage.running_jobs}</Table.Td><Table.Td>{usage.queued_jobs}</Table.Td></Table.Tr>
                  ))}</Table.Tbody>
                </Table>
              )}
            </Stack>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>

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
              onChange={(event) => {
                const checked = event.currentTarget.checked;
                setSettingsDraft((current) => ({
                  max_gpu_slots: current?.max_gpu_slots ?? 1,
                  only_gpu: checked,
                }));
              }}
            />
            <Button onClick={handleSaveSettings} loading={savingSettings} disabled={!settingsDraft}>
              Save settings
            </Button>
          </Group>
        </Stack>
      </Paper>

      <Paper withBorder p="md" radius="sm">
        <Stack gap="md">
          <Group justify="space-between">
            <Text fw={600}>Job visibility</Text>
            <SegmentedControl
              value={scope}
              onChange={(value) => setScope(value as 'project' | 'all')}
              data={[{ value: 'project', label: 'Current project' }, { value: 'all', label: 'All projects' }]}
            />
          </Group>
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
                { value: 'heatmap', label: 'Heatmap' },
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
                  {scope === 'all' && <Table.Th>Project</Table.Th>}
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
                  const key = jobKey(job);
                  const queueIndex = queueIndexByKey.get(key);
                  const movable = run.status === 'queued' && queueIndex !== undefined;
                  const rowBusy =
                    rowActions.isPending(`abort:${key}`) ||
                    rowActions.isPending(`restart:${key}`) ||
                    rowActions.isPending(`delete:${key}`) ||
                    rowActions.isPending(`move-up:${key}`) ||
                    rowActions.isPending(`move-down:${key}`);
                  return (
                    <Table.Tr key={key}>
                      <Table.Td>
                        <Badge
                          color={job.kind === 'train' ? 'blue' : job.kind === 'heatmap' ? 'teal' : 'grape'}
                          variant="light"
                        >
                          {job.kind === 'train' ? 'Training' : job.kind === 'heatmap' ? 'Heatmap' : 'Inference'}
                        </Badge>
                      </Table.Td>
                      {scope === 'all' && <Table.Td><Badge variant="outline">{job.project_name}</Badge></Table.Td>}
                      <Table.Td>{jobName(job)}</Table.Td>
                      <Table.Td>
                        <Text size="sm">{jobMethodType(job)}</Text>
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
                          {movable && (
                            <>
                              <Tooltip label="Move up">
                                <ActionIcon
                                  variant="subtle"
                                  loading={rowActions.isPending(`move-up:${key}`)}
                                  disabled={rowBusy || queueIndex === 0}
                                  onClick={() => handleMove(job, 'up')}
                                >
                                  <ArrowUp size={18} />
                                </ActionIcon>
                              </Tooltip>
                              <Tooltip label="Move down">
                                <ActionIcon
                                  variant="subtle"
                                  loading={rowActions.isPending(`move-down:${key}`)}
                                  disabled={rowBusy || queueIndex === queuedJobs.length - 1}
                                  onClick={() => handleMove(job, 'down')}
                                >
                                  <ArrowDown size={18} />
                                </ActionIcon>
                              </Tooltip>
                            </>
                          )}
                          {abortable && (
                            <Tooltip label="Abort">
                              <ActionIcon
                                color="orange"
                                variant="subtle"
                                loading={rowActions.isPending(`abort:${key}`)}
                                disabled={rowBusy && !rowActions.isPending(`abort:${key}`)}
                                onClick={() => handleAbort(job)}
                              >
                                <StopCircle size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                          {terminal && job.kind !== 'heatmap' && (
                            <Tooltip label="Restart">
                              <ActionIcon
                                variant="subtle"
                                loading={rowActions.isPending(`restart:${key}`)}
                                disabled={rowBusy && !rowActions.isPending(`restart:${key}`)}
                                onClick={() => handleRestart(job)}
                              >
                                <RotateCcw size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                          {run.status !== 'running' && (
                            <Tooltip label="Remove">
                              <ActionIcon
                                color="red"
                                variant="subtle"
                                loading={rowActions.isPending(`delete:${key}`)}
                                disabled={rowBusy && !rowActions.isPending(`delete:${key}`)}
                                onClick={() => handleDelete(job)}
                              >
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
            <Button
              variant="default"
              color="red"
              leftSection={<Trash2 size={16} />}
              loading={clearingHeatmaps}
              disabled={heatmaps.length === 0 || clearingHeatmaps}
              onClick={handleClearHeatmaps}
            >
              {clearingHeatmaps ? 'Clearing…' : 'Clear heatmaps'}
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
