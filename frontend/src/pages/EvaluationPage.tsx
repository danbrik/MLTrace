import {
  Alert,
  Badge,
  Button,
  Collapse,
  Group,
  Loader,
  MultiSelect,
  NumberInput,
  Paper,
  ScrollArea,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import {
  ArrowLeft,
  Calculator,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  FileCheck2,
  Filter,
  Plus,
  RefreshCw,
  Save,
  Settings2,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  calculateModelEvaluation,
  createModelEvaluation,
  deleteModelEvaluation,
  duplicateModelEvaluation,
  evaluationLabelSetCsvUrl,
  finalizeModelEvaluation,
  getEvaluationScorePreview,
  getModelEvaluation,
  listEvaluationLabelSets,
  listEvaluationProfiles,
  listModelEvaluations,
  listTestingRuns,
  listTrainingDatasets,
  updateModelEvaluation,
} from '../api';
import { StepCard, STEP_COLORS } from '../components/StepCard';
import { EvaluationProfileManager, EvaluationLabelSetManager } from '../evaluation/EvaluationResourceManagers';
import { EvaluationResults } from '../evaluation/EvaluationResults';
import { EvaluationRunPicker } from '../evaluation/EvaluationRunPicker';
import { EvaluationTimeline } from '../evaluation/EvaluationTimeline';
import { projectEvaluationDraft, projectedCalculationStatus } from '../evaluation/draftFreshness';
import {
  calculationStatusColor,
  calculationStatusLabel,
  canFinalizeEvaluation,
  DEFAULT_EVALUATION_QUANTILE,
  rangesOverlap,
  scoreSeriesForRun,
  statusIsStale,
} from '../evaluation/helpers';
import type {
  EvaluationLabelEvent,
  EvaluationLabelSet,
  EvaluationNormalWindowOverride,
  EvaluationProfile,
  EvaluationScorePreview,
  EvaluationTimeRange,
  ModelEvaluation,
  ModelEvaluationPayload,
  TestingRun,
  TrainingDataset,
} from '../types';

const EVALUATION_PREVIEW_POINTS = 8000;

type EvaluationView = 'drafts' | 'finalized';
type SourceRole = 'evaluation' | 'reference' | 'calibration';
type SavedStaleFilter = 'all' | 'current' | 'stale';
type ProfileOverrideField = 'normal_window_duration_seconds' | 'normal_window_buffer_seconds' | 'drift_window_seconds' | 'false_alarm_horizon_seconds' | 'anticipation_seconds' | 'epsilon';

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function emptyDraft(): ModelEvaluationPayload {
  return {
    name: '',
    evaluation_testing_run_id: null,
    evaluation_start_timestamp: null,
    evaluation_end_timestamp: null,
    reference_testing_run_id: null,
    reference_start_timestamp: null,
    reference_end_timestamp: null,
    calibration_testing_run_id: null,
    calibration_start_timestamp: null,
    calibration_end_timestamp: null,
    score_series: 'score',
    label_set_id: null,
    profile_id: null,
    selected_categories: [],
    normal_window_overrides: {},
    profile_overrides: {},
    active_quantile: DEFAULT_EVALUATION_QUANTILE,
  };
}

function payloadFromEvaluation(evaluation: ModelEvaluation): ModelEvaluationPayload {
  return {
    name: evaluation.name,
    evaluation_testing_run_id: evaluation.evaluation_testing_run_id,
    evaluation_start_timestamp: evaluation.evaluation_start_timestamp,
    evaluation_end_timestamp: evaluation.evaluation_end_timestamp,
    reference_testing_run_id: evaluation.reference_testing_run_id,
    reference_start_timestamp: evaluation.reference_start_timestamp,
    reference_end_timestamp: evaluation.reference_end_timestamp,
    calibration_testing_run_id: evaluation.calibration_testing_run_id,
    calibration_start_timestamp: evaluation.calibration_start_timestamp,
    calibration_end_timestamp: evaluation.calibration_end_timestamp,
    score_series: evaluation.score_series,
    label_set_id: evaluation.label_set_id,
    profile_id: evaluation.profile_id,
    selected_categories: evaluation.selected_categories ?? [],
    normal_window_overrides: evaluation.normal_window_overrides ?? {},
    profile_overrides: evaluation.profile_overrides ?? {},
    active_quantile: evaluation.active_quantile ?? DEFAULT_EVALUATION_QUANTILE,
  };
}

function roleRunId(draft: ModelEvaluationPayload, role: SourceRole): number | null {
  return draft[`${role}_testing_run_id`];
}

function roleRange(draft: ModelEvaluationPayload, role: SourceRole): EvaluationTimeRange {
  return {
    start_timestamp: draft[`${role}_start_timestamp`] ?? '',
    end_timestamp: draft[`${role}_end_timestamp`] ?? '',
  };
}

function withRoleRun(draft: ModelEvaluationPayload, role: SourceRole, runId: number | null): ModelEvaluationPayload {
  return {
    ...draft,
    [`${role}_testing_run_id`]: runId,
    [`${role}_start_timestamp`]: null,
    [`${role}_end_timestamp`]: null,
  };
}

function withRoleRange(draft: ModelEvaluationPayload, role: SourceRole, range: EvaluationTimeRange): ModelEvaluationPayload {
  return {
    ...draft,
    [`${role}_start_timestamp`]: range.start_timestamp || null,
    [`${role}_end_timestamp`]: range.end_timestamp || null,
  };
}

function completeRole(draft: ModelEvaluationPayload, role: SourceRole): boolean {
  return roleRunId(draft, role) != null
    && Boolean(draft[`${role}_start_timestamp`])
    && Boolean(draft[`${role}_end_timestamp`]);
}

function subtractSeconds(timestamp: string, seconds: number): string {
  if (!timestamp) return '';
  const parsed = new Date(`${timestamp.slice(0, 19)}Z`).getTime();
  if (!Number.isFinite(parsed)) return '';
  return new Date(parsed - seconds * 1000).toISOString().slice(0, 19);
}

function snapshotEvents(evaluation: ModelEvaluation | null): EvaluationLabelEvent[] {
  return evaluation?.label_snapshot?.events ?? [];
}

function downloadLabelSnapshot(events: EvaluationLabelEvent[], name: string) {
  const quote = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const header = ['event_id', 'type', 'name', 'category', 'start_timestamp', 'end_timestamp', 'notes'];
  const rows = events.map((event) => [event.event_id, event.type, event.name, event.category, event.start_timestamp, event.end_timestamp, event.notes]);
  const blob = new Blob([[header, ...rows].map((row) => row.map(quote).join(',')).join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${name.trim().replace(/[^a-z0-9_-]+/gi, '-') || 'evaluation'}-ground-truth.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function EvaluationStatusBadge({ status }: { status: string }) {
  return <Badge color={calculationStatusColor(status)} variant="light">{calculationStatusLabel(status)}</Badge>;
}

function StageAction({
  stage,
  label,
  status,
  error,
  disabled,
  calculating,
  onCalculate,
}: {
  stage: 'separation' | 'drift' | 'detection';
  label: string;
  status: string;
  error?: string | null;
  disabled: boolean;
  calculating: string | null;
  onCalculate: (stage: 'separation' | 'drift' | 'detection') => void;
}) {
  return (
    <Paper withBorder p="sm">
      <Stack gap="xs">
        <Group justify="space-between"><Text fw={600}>{label}</Text><EvaluationStatusBadge status={status} /></Group>
        <Text size="xs" c="dimmed">Runs independently; an error here leaves other results available.</Text>
        {error && <Alert color="red" p="xs">{error}</Alert>}
        <Button
          variant="light"
          color={stage === 'separation' ? 'violet' : stage === 'drift' ? 'teal' : 'orange'}
          leftSection={<Calculator size={15} />}
          loading={calculating === stage}
          disabled={disabled || Boolean(calculating)}
          onClick={() => onCalculate(stage)}
        >
          {status === 'not_calculated' ? 'Calculate' : 'Recalculate'} {stage === 'separation' ? 'A' : stage === 'drift' ? 'B' : 'C'}
        </Button>
      </Stack>
    </Paper>
  );
}

export function EvaluationPage({ active }: { active: boolean }) {
  const [testingRuns, setTestingRuns] = useState<TestingRun[]>([]);
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [profiles, setProfiles] = useState<EvaluationProfile[]>([]);
  const [labelSets, setLabelSets] = useState<EvaluationLabelSet[]>([]);
  const [evaluations, setEvaluations] = useState<ModelEvaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<EvaluationView>('drafts');
  const [selectedEvaluation, setSelectedEvaluation] = useState<ModelEvaluation | null>(null);
  const [draft, setDraft] = useState<ModelEvaluationPayload>(emptyDraft);
  const [creating, setCreating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [calculating, setCalculating] = useState<string | null>(null);
  const [previewRole, setPreviewRole] = useState<SourceRole>('evaluation');
  const [previews, setPreviews] = useState<Partial<Record<SourceRole, EvaluationScorePreview>>>({});
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [profileManagerOpen, setProfileManagerOpen] = useState(false);
  const [labelManagerOpen, setLabelManagerOpen] = useState(false);
  const [suggestedEvent, setSuggestedEvent] = useState<EvaluationLabelEvent | null>(null);
  const [overridesOpen, setOverridesOpen] = useState(false);
  const [profileOverridesOpen, setProfileOverridesOpen] = useState(false);
  const [overrideEventId, setOverrideEventId] = useState<string | null>(null);
  const [overrideRange, setOverrideRange] = useState<EvaluationTimeRange>({ start_timestamp: '', end_timestamp: '' });
  const [savedFiltersOpen, setSavedFiltersOpen] = useState(false);
  const [savedQuery, setSavedQuery] = useState('');
  const [savedStale, setSavedStale] = useState<SavedStaleFilter>('all');
  const [savedCategories, setSavedCategories] = useState<string[]>([]);
  const [savedScores, setSavedScores] = useState<string[]>([]);
  const [createdFrom, setCreatedFrom] = useState('');
  const [createdTo, setCreatedTo] = useState('');

  const refreshProfiles = useCallback(async () => setProfiles(await listEvaluationProfiles()), []);
  const refreshLabelSets = useCallback(async () => setLabelSets(await listEvaluationLabelSets()), []);
  const refreshEvaluations = useCallback(async () => setEvaluations(await listModelEvaluations()), []);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    try {
      const [nextRuns, nextDatasets, nextProfiles, nextLabelSets, nextEvaluations] = await Promise.all([
        listTestingRuns(), listTrainingDatasets(), listEvaluationProfiles(), listEvaluationLabelSets(), listModelEvaluations(),
      ]);
      setTestingRuns(nextRuns);
      setDatasets(nextDatasets);
      setProfiles(nextProfiles);
      setLabelSets(nextLabelSets);
      setEvaluations(nextEvaluations);
    } catch (error) {
      notifyError('Could not load evaluation workspace', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (active) void refreshAll();
  }, [active, refreshAll]);

  const evaluationRun = testingRuns.find((run) => run.id === draft.evaluation_testing_run_id) ?? null;
  const selectedLabelSet = labelSets.find((labelSet) => labelSet.id === draft.label_set_id) ?? null;
  const selectedProfile = profiles.find((profile) => profile.id === draft.profile_id) ?? null;
  const readonly = selectedEvaluation?.status === 'finalized';
  const draftProjection = useMemo(
    () => projectEvaluationDraft(selectedEvaluation ? payloadFromEvaluation(selectedEvaluation) : null, draft),
    [draft, selectedEvaluation],
  );
  const projectedEvaluation = useMemo(() => {
    if (!selectedEvaluation || readonly || draftProjection.changedStages.size === 0) return selectedEvaluation;
    return {
      ...selectedEvaluation,
      separation_status: projectedCalculationStatus(selectedEvaluation.separation_status, draftProjection.changedStages.has('separation')),
      separation_error: draftProjection.changedStages.has('separation') ? null : selectedEvaluation.separation_error,
      drift_status: projectedCalculationStatus(selectedEvaluation.drift_status, draftProjection.changedStages.has('drift')),
      drift_error: draftProjection.changedStages.has('drift') ? null : selectedEvaluation.drift_error,
      detection_status: projectedCalculationStatus(selectedEvaluation.detection_status, draftProjection.changedStages.has('detection')),
      detection_error: draftProjection.changedStages.has('detection') ? null : selectedEvaluation.detection_error,
    };
  }, [draftProjection.changedStages, readonly, selectedEvaluation]);
  const displayedProfile = readonly && selectedEvaluation?.profile_snapshot
    ? { ...selectedEvaluation.profile_snapshot, ...selectedEvaluation.profile_overrides }
    : selectedProfile;
  const displayedEvents = selectedEvaluation?.status === 'finalized'
    ? snapshotEvents(selectedEvaluation)
    : selectedLabelSet?.events ?? snapshotEvents(selectedEvaluation);
  const targetEvents = displayedEvents.filter((event) => event.type === 'target');
  const currentRoleRunId = roleRunId(draft, previewRole);
  const currentRoleRange = roleRange(draft, previewRole);
  const currentPreview = previews[previewRole] ?? null;
  const isEditorOpen = creating || selectedEvaluation !== null;

  useEffect(() => {
    let cancelled = false;
    if (!active || currentRoleRunId == null) {
      setPreviewError(null);
      return undefined;
    }
    setPreviewLoading(true);
    setPreviewError(null);
    getEvaluationScorePreview(currentRoleRunId, { score_series: draft.score_series, max_points: EVALUATION_PREVIEW_POINTS })
      .then((preview) => {
        if (cancelled) return;
        setPreviews((current) => ({ ...current, [previewRole]: preview }));
        if (!currentRoleRange.start_timestamp && !currentRoleRange.end_timestamp && preview.start_timestamp && preview.end_timestamp && !readonly) {
          setDraft((current) => withRoleRange(current, previewRole, { start_timestamp: preview.start_timestamp!, end_timestamp: preview.end_timestamp! }));
        }
      })
      .catch((error) => { if (!cancelled) setPreviewError(error instanceof Error ? error.message : 'Could not load score preview'); })
      .finally(() => { if (!cancelled) setPreviewLoading(false); });
    return () => { cancelled = true; };
  }, [active, currentRoleRange.end_timestamp, currentRoleRange.start_timestamp, currentRoleRunId, draft.score_series, previewRole, readonly]);

  function openEvaluation(evaluation: ModelEvaluation) {
    setSelectedEvaluation(evaluation);
    setDraft(payloadFromEvaluation(evaluation));
    setCreating(false);
    setPreviewRole('evaluation');
    setPreviews({});
    setPreviewError(null);
  }

  function beginNew() {
    setSelectedEvaluation(null);
    setDraft(emptyDraft());
    setCreating(true);
    setPreviewRole('evaluation');
    setPreviews({});
  }

  function closeEditor() {
    setCreating(false);
    setSelectedEvaluation(null);
    setDraft(emptyDraft());
    setPreviews({});
  }

  function replaceEvaluation(updated: ModelEvaluation, syncDraft = true) {
    setSelectedEvaluation(updated);
    if (syncDraft) setDraft(payloadFromEvaluation(updated));
    setEvaluations((current) => {
      const exists = current.some((evaluation) => evaluation.id === updated.id);
      return exists ? current.map((evaluation) => evaluation.id === updated.id ? updated : evaluation) : [updated, ...current];
    });
  }

  async function persistDraft(): Promise<ModelEvaluation | null> {
    if (!draft.name.trim() || draft.evaluation_testing_run_id == null) {
      notifications.show({ color: 'yellow', title: 'Complete step 1', message: 'Give the evaluation a name and select its evaluation inference run.' });
      return null;
    }
    setSaving(true);
    try {
      const payload = { ...draft, name: draft.name.trim() };
      const saved = selectedEvaluation
        ? await updateModelEvaluation(selectedEvaluation.id, payload)
        : await createModelEvaluation(payload);
      replaceEvaluation(saved);
      setCreating(false);
      notifications.show({ color: 'green', title: 'Draft saved', message: saved.name });
      return saved;
    } catch (error) {
      notifyError('Could not save evaluation draft', error);
      return null;
    } finally {
      setSaving(false);
    }
  }

  async function calculate(stage: 'separation' | 'drift' | 'detection') {
    const saved = await persistDraft();
    if (!saved) return;
    setCalculating(stage);
    try {
      const calculated = await calculateModelEvaluation(saved.id, stage);
      replaceEvaluation(calculated);
      notifications.show({ color: 'green', title: `Step ${stage === 'separation' ? 'A' : stage === 'drift' ? 'B' : 'C'} calculated`, message: 'The other calculation steps were left unchanged.' });
    } catch (error) {
      notifyError(`Could not calculate ${stage}`, error);
      try { replaceEvaluation(await getModelEvaluation(saved.id)); } catch { /* the original API error is more useful */ }
    } finally {
      setCalculating(null);
    }
  }

  async function finalize() {
    if (!selectedEvaluation) return;
    const saved = await persistDraft();
    if (!saved) return;
    setSaving(true);
    try {
      const finalized = await finalizeModelEvaluation(saved.id);
      replaceEvaluation(finalized);
      setView('finalized');
      notifications.show({ color: 'green', title: 'Evaluation finalized', message: 'The reproducible snapshot is now immutable.' });
    } catch (error) {
      notifyError('Could not finalize evaluation', error);
    } finally {
      setSaving(false);
    }
  }

  async function duplicate(evaluation: ModelEvaluation) {
    try {
      const copy = await duplicateModelEvaluation(evaluation.id);
      setView('drafts');
      replaceEvaluation(copy);
      notifications.show({ color: 'green', title: 'Draft duplicated', message: copy.name });
    } catch (error) {
      notifyError('Could not duplicate evaluation', error);
    }
  }

  const savedCategoryOptions = useMemo(() => [...new Set(evaluations.flatMap((evaluation) => (
    evaluation.selected_categories.length ? evaluation.selected_categories : snapshotEvents(evaluation).map((event) => event.category || 'Uncategorized')
  )))].sort().map((value) => ({ value, label: value })), [evaluations]);
  const savedScoreOptions = useMemo(() => [...new Set(evaluations.map((evaluation) => evaluation.score_series))].sort().map((value) => ({ value, label: value })), [evaluations]);
  const filteredEvaluations = useMemo(() => evaluations.filter((evaluation) => {
    if (evaluation.status !== (view === 'drafts' ? 'draft' : 'finalized')) return false;
    if (savedQuery.trim() && !evaluation.name.toLowerCase().includes(savedQuery.trim().toLowerCase())) return false;
    const stale = statusIsStale(evaluation.separation_status) || statusIsStale(evaluation.drift_status) || statusIsStale(evaluation.detection_status);
    if (savedStale === 'stale' && !stale) return false;
    if (savedStale === 'current' && stale) return false;
    const categories = evaluation.selected_categories.length ? evaluation.selected_categories : snapshotEvents(evaluation).map((event) => event.category || 'Uncategorized');
    if (savedCategories.length && !categories.some((category) => savedCategories.includes(category))) return false;
    if (savedScores.length && !savedScores.includes(evaluation.score_series)) return false;
    if (createdFrom && evaluation.created_at < createdFrom) return false;
    if (createdTo && evaluation.created_at > `${createdTo}T23:59:59`) return false;
    return true;
  }), [createdFrom, createdTo, evaluations, savedCategories, savedQuery, savedScores, savedStale, view]);

  const sameRunOverlap = useMemo(() => {
    const roles: SourceRole[] = ['evaluation', 'reference', 'calibration'];
    const overlaps: string[] = [];
    roles.forEach((left, index) => roles.slice(index + 1).forEach((right) => {
      if (roleRunId(draft, left) != null && roleRunId(draft, left) === roleRunId(draft, right)) {
        const first = roleRange(draft, left); const second = roleRange(draft, right);
        if (rangesOverlap(first.start_timestamp, first.end_timestamp, second.start_timestamp, second.end_timestamp)) overlaps.push(`${left} / ${right}`);
      }
    }));
    return overlaps;
  }, [draft]);

  const aReady = completeRole(draft, 'evaluation') && draft.label_set_id != null && draft.profile_id != null;
  const bReady = aReady && completeRole(draft, 'reference') && sameRunOverlap.length === 0;
  const cReady = aReady && completeRole(draft, 'calibration') && sameRunOverlap.length === 0;

  function selectEvaluationRun(runId: number | null) {
    const previous = testingRuns.find((run) => run.id === draft.evaluation_testing_run_id);
    const next = testingRuns.find((run) => run.id === runId);
    setDraft((current) => ({
      ...withRoleRun(withRoleRun(withRoleRun(current, 'evaluation', runId), 'reference', null), 'calibration', null),
      score_series: 'score',
      label_set_id: previous?.training_dataset_id === next?.training_dataset_id ? current.label_set_id : null,
      selected_categories: previous?.training_dataset_id === next?.training_dataset_id ? current.selected_categories : [],
      normal_window_overrides: {},
    }));
    setPreviews({});
    setPreviewRole('evaluation');
  }

  function setRoleSource(role: 'reference' | 'calibration', runId: number | null) {
    setDraft((current) => withRoleRun(current, role, runId));
    setPreviews((current) => ({ ...current, [role]: undefined }));
    setPreviewRole(role);
  }

  function addSuggestedEvent(type: 'target' | 'exclusion', range: EvaluationTimeRange) {
    if (!selectedLabelSet) {
      notifications.show({ color: 'yellow', title: 'Select or create a label set', message: 'Intervals belong to a reusable ground-truth set.' });
    }
    setSuggestedEvent({ event_id: crypto.randomUUID(), type, name: '', category: '', start_timestamp: range.start_timestamp, end_timestamp: range.end_timestamp, notes: '' });
    setLabelManagerOpen(true);
  }

  function selectOverrideEvent(eventId: string | null) {
    setOverrideEventId(eventId);
    const existing = eventId ? draft.normal_window_overrides[eventId] : null;
    const event = targetEvents.find((candidate) => candidate.event_id === eventId);
    if (existing) {
      setOverrideRange(existing);
    } else if (event && selectedProfile) {
      const buffer = draft.profile_overrides.normal_window_buffer_seconds ?? selectedProfile.normal_window_buffer_seconds;
      const duration = draft.profile_overrides.normal_window_duration_seconds ?? selectedProfile.normal_window_duration_seconds;
      const end = subtractSeconds(event.start_timestamp, buffer);
      setOverrideRange({ start_timestamp: subtractSeconds(end, duration), end_timestamp: end });
    } else {
      setOverrideRange({ start_timestamp: '', end_timestamp: '' });
    }
  }

  function setProfileOverride(field: ProfileOverrideField, value: string | number) {
    setDraft((current) => {
      const next = { ...current.profile_overrides };
      if (typeof value === 'number' && Number.isFinite(value)) next[field] = value;
      else delete next[field];
      return { ...current, profile_overrides: next };
    });
  }

  if (loading && evaluations.length === 0) return <Group justify="center" py="xl"><Loader /><Text>Loading evaluation workspace…</Text></Group>;

  return (
    <Stack gap="lg">
      <Group justify="space-between" align="start">
        <div><Title order={2}>Evaluation</Title><Text c="dimmed">Calculate eight reproducible principal metrics for one model artifact across event separation, drift and detection performance.</Text></div>
        <Group>
          <Button variant="default" leftSection={<Settings2 size={16} />} onClick={() => setProfileManagerOpen(true)}>Profiles</Button>
          <Button variant="default" leftSection={<FileCheck2 size={16} />} onClick={() => setLabelManagerOpen(true)}>Ground truth</Button>
          <Button variant="default" leftSection={<RefreshCw size={16} />} onClick={() => void refreshAll()}>Refresh</Button>
          {!isEditorOpen && <Button leftSection={<Plus size={16} />} onClick={beginNew}>New evaluation</Button>}
        </Group>
      </Group>

      {!isEditorOpen && (
        <>
          <Group justify="space-between">
            <SegmentedControl value={view} onChange={(value) => setView(value as EvaluationView)} data={[{ value: 'drafts', label: 'Drafts & calculation' }, { value: 'finalized', label: 'Finalized evaluations' }]} />
            <Group><Badge variant="light">{filteredEvaluations.length} evaluations</Badge><Button variant="default" size="compact-sm" leftSection={<Filter size={14} />} rightSection={savedFiltersOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />} onClick={() => setSavedFiltersOpen((current) => !current)}>Filters</Button></Group>
          </Group>
          <Collapse in={savedFiltersOpen}>
            <Paper withBorder p="md"><Stack><SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}><TextInput label="Search" value={savedQuery} onChange={(event) => setSavedQuery(event.currentTarget.value)} placeholder="Evaluation name" /><Select label="Freshness" value={savedStale} onChange={(value) => setSavedStale((value ?? 'all') as SavedStaleFilter)} data={[{ value: 'all', label: 'All' }, { value: 'current', label: 'No stale stages' }, { value: 'stale', label: 'Contains stale stage' }]} /><MultiSelect label="Categories" data={savedCategoryOptions} value={savedCategories} onChange={setSavedCategories} searchable clearable /><MultiSelect label="Score series" data={savedScoreOptions} value={savedScores} onChange={setSavedScores} clearable /></SimpleGrid><SimpleGrid cols={{ base: 1, sm: 2 }}><TextInput type="date" label="Created from" value={createdFrom} onChange={(event) => setCreatedFrom(event.currentTarget.value)} /><TextInput type="date" label="Created to" value={createdTo} onChange={(event) => setCreatedTo(event.currentTarget.value)} /></SimpleGrid></Stack></Paper>
          </Collapse>
          <Paper withBorder>
            <ScrollArea>
              <Table striped highlightOnHover miw={980}>
                <Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Score</Table.Th><Table.Th>A · Separation</Table.Th><Table.Th>B · Drift</Table.Th><Table.Th>C · Detection</Table.Th><Table.Th>Updated</Table.Th><Table.Th /></Table.Tr></Table.Thead>
                <Table.Tbody>
                  {filteredEvaluations.map((evaluation) => <Table.Tr key={evaluation.id}><Table.Td><Text fw={600}>{evaluation.name}</Text><Text size="xs" c="dimmed">#{evaluation.id}{evaluation.status === 'finalized' && evaluation.finalized_at ? ` · finalized ${evaluation.finalized_at.replace('T', ' ')}` : ''}</Text></Table.Td><Table.Td><Badge variant="outline">{evaluation.score_series}</Badge></Table.Td><Table.Td><EvaluationStatusBadge status={evaluation.separation_status} /></Table.Td><Table.Td><EvaluationStatusBadge status={evaluation.drift_status} /></Table.Td><Table.Td><EvaluationStatusBadge status={evaluation.detection_status} /></Table.Td><Table.Td>{evaluation.updated_at.replace('T', ' ')}</Table.Td><Table.Td><Group justify="flex-end" gap="xs"><Button size="compact-xs" onClick={() => openEvaluation(evaluation)}>{evaluation.status === 'finalized' ? 'View' : 'Open'}</Button><Button size="compact-xs" variant="light" leftSection={<Copy size={12} />} onClick={() => void duplicate(evaluation)}>Duplicate</Button><Button size="compact-xs" variant="subtle" color="red" leftSection={<Trash2 size={12} />} onClick={async () => { if (!window.confirm(`Delete evaluation “${evaluation.name}”?`)) return; try { await deleteModelEvaluation(evaluation.id); await refreshEvaluations(); } catch (error) { notifyError('Could not delete evaluation', error); } }}>Delete</Button></Group></Table.Td></Table.Tr>)}
                  {filteredEvaluations.length === 0 && <Table.Tr><Table.Td colSpan={7}><Text ta="center" c="dimmed" py="xl">No evaluations match this view and its filters.</Text></Table.Td></Table.Tr>}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Paper>
        </>
      )}

      {isEditorOpen && (
        <>
          <Group justify="space-between">
            <Button variant="subtle" color="gray" leftSection={<ArrowLeft size={16} />} onClick={closeEditor}>Back to {view === 'drafts' ? 'drafts' : 'finalized evaluations'}</Button>
            <Group>
              {!readonly && draftProjection.dirty && <Badge color="yellow" variant="light">Unsaved changes</Badge>}
              {selectedEvaluation && <Button variant="light" leftSection={<Copy size={15} />} onClick={() => void duplicate(selectedEvaluation)}>Duplicate</Button>}
              {!readonly && <Button leftSection={<Save size={15} />} loading={saving} onClick={() => void persistDraft()}>Save draft</Button>}
            </Group>
          </Group>
          {readonly && <Alert color="green" icon={<CheckCircle2 size={18} />} title="Immutable evaluation snapshot">This evaluation was finalized at {selectedEvaluation?.finalized_at?.replace('T', ' ') ?? 'an unknown time'}. Duplicate it to change inputs or recalculate metrics.</Alert>}
          {!readonly && selectedEvaluation && draftProjection.dirty && <Alert color="yellow" title="Draft has unsaved changes">Affected calculated stages are shown as stale immediately. Save before finalizing; calculations always save the draft first.</Alert>}

          <StepCard index={1} title="Name, model artifact and score" subtitle="Choose exactly one finished evaluation inference run and one stored score series." color={STEP_COLORS[0]} complete={Boolean(draft.name.trim() && draft.evaluation_testing_run_id)}>
            <TextInput label="Evaluation name" value={draft.name} onChange={(event) => setDraft((current) => ({ ...current, name: event.currentTarget.value }))} disabled={readonly} required />
            <EvaluationRunPicker runs={testingRuns} value={draft.evaluation_testing_run_id} onChange={selectEvaluationRun} label="Evaluation inference run" description="The selected model artifact defines which reference and calibration runs are compatible." disabled={readonly} />
            <Select label="Score series" data={scoreSeriesForRun(evaluationRun)} value={draft.score_series} onChange={(value) => { setDraft((current) => ({ ...current, score_series: value ?? 'score' })); setPreviews({}); }} disabled={readonly || !evaluationRun} />
            {evaluationRun && !evaluationRun.artifact_signature && <Alert color="yellow">This legacy inference run has no artifact signature. It can be inspected, but compatible reference/calibration sources cannot be proven and therefore are not offered.</Alert>}
          </StepCard>

          <StepCard index={2} title="Evaluation period" subtitle="Drag horizontally or enter dataset-local 24-hour timestamps. Zoom, pan and the range slider never alter calculation inputs." color={STEP_COLORS[1]} complete={completeRole(draft, 'evaluation')}>
            <EvaluationTimeline preview={previews.evaluation ?? null} loading={previewRole === 'evaluation' && previewLoading} error={previewRole === 'evaluation' ? previewError : null} range={roleRange(draft, 'evaluation')} onRangeChange={(range) => setDraft((current) => withRoleRange(current, 'evaluation', range))} events={displayedEvents} onCreateEvent={addSuggestedEvent} disabled={readonly || !evaluationRun} />
          </StepCard>

          <StepCard index={3} title="Ground truth" subtitle="Use a reusable, versioned label set tied to the evaluation inference dataset." color={STEP_COLORS[2]} complete={Boolean(selectedLabelSet || selectedEvaluation?.label_snapshot)} action={!readonly && <Button size="compact-sm" variant="light" onClick={() => setLabelManagerOpen(true)}>Manage label sets</Button>}>
            <Group align="end" grow>
              {readonly ? <TextInput label="Label snapshot" value={`${selectedEvaluation?.label_snapshot?.name ?? 'Ground truth'} · v${selectedEvaluation?.label_snapshot?.version ?? '—'} · ${targetEvents.length} targets`} disabled /> : <Select label="Label set" placeholder="Select ground truth" data={labelSets.filter((set) => !evaluationRun || set.training_dataset_id === evaluationRun.training_dataset_id).map((set) => ({ value: String(set.id), label: `${set.name} · v${set.version} · ${set.events.filter((event) => event.type === 'target').length} targets` }))} value={draft.label_set_id == null ? null : String(draft.label_set_id)} onChange={(value) => { const id = value ? Number(value) : null; const set = labelSets.find((candidate) => candidate.id === id); setDraft((current) => ({ ...current, label_set_id: id, selected_categories: set?.categories ?? [], normal_window_overrides: {} })); }} searchable clearable disabled={!evaluationRun} />}
              {readonly ? <Button variant="default" leftSection={<Download size={15} />} onClick={() => downloadLabelSnapshot(displayedEvents, selectedEvaluation?.name ?? 'evaluation')}>Export snapshot CSV</Button> : selectedLabelSet && <Button component="a" href={evaluationLabelSetCsvUrl(selectedLabelSet.id)} download variant="default" leftSection={<Download size={15} />}>Export CSV</Button>}
            </Group>
            {readonly ? <Paper withBorder p="xs"><Text size="xs" c="dimmed" mb={4}>Included target categories</Text><Group gap="xs">{(draft.selected_categories.length ? draft.selected_categories : [...new Set(targetEvents.map((event) => event.category || 'Uncategorized'))]).map((category) => <Badge key={category} variant="light">{category}</Badge>)}</Group></Paper> : selectedLabelSet && <MultiSelect label="Included target categories" description="Leave empty to use every target category." data={selectedLabelSet.categories.map((category) => ({ value: category, label: category }))} value={draft.selected_categories} onChange={(values) => setDraft((current) => ({ ...current, selected_categories: values, normal_window_overrides: {} }))} searchable clearable />}
            <ScrollArea h={220}><Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Type</Table.Th><Table.Th>Name</Table.Th><Table.Th>Category</Table.Th><Table.Th>Start</Table.Th><Table.Th>End (inclusive)</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{displayedEvents.map((event) => <Table.Tr key={event.event_id}><Table.Td><Badge color={event.type === 'target' ? 'red' : 'gray'} variant="light">{event.type}</Badge></Table.Td><Table.Td>{event.name || '—'}</Table.Td><Table.Td>{event.category || '—'}</Table.Td><Table.Td>{event.start_timestamp.replace('T', ' ')}</Table.Td><Table.Td>{event.end_timestamp.replace('T', ' ')}</Table.Td></Table.Tr>)}{displayedEvents.length === 0 && <Table.Tr><Table.Td colSpan={5}><Text ta="center" c="dimmed">No ground-truth intervals selected.</Text></Table.Td></Table.Tr>}</Table.Tbody></Table></ScrollArea>
          </StepCard>

          <StepCard index={4} title="Normal reference and calibration roles" subtitle="Sources may reuse one compatible run, but all ranges on that run must be pairwise disjoint." color={STEP_COLORS[3]} complete={completeRole(draft, 'reference') && completeRole(draft, 'calibration') && sameRunOverlap.length === 0}>
            <SimpleGrid cols={{ base: 1, lg: 2 }}>
              <EvaluationRunPicker runs={testingRuns} value={draft.reference_testing_run_id} onChange={(value) => setRoleSource('reference', value)} compatibleWith={evaluationRun} label="Reference inference run" description="Only explicitly normal reference scores are used for drift." disabled={readonly || !evaluationRun} />
              <EvaluationRunPicker runs={testingRuns} value={draft.calibration_testing_run_id} onChange={(value) => setRoleSource('calibration', value)} compatibleWith={evaluationRun} label="Calibration inference run" description="Only explicitly normal calibration scores define thresholds." disabled={readonly || !evaluationRun} />
            </SimpleGrid>
            <Group justify="space-between"><Text fw={600} size="sm">Edit role range</Text><SegmentedControl value={previewRole} onChange={(value) => setPreviewRole(value as SourceRole)} data={[{ value: 'evaluation', label: 'Evaluation' }, { value: 'reference', label: 'Reference' }, { value: 'calibration', label: 'Calibration' }]} /></Group>
            <EvaluationTimeline preview={currentPreview} loading={previewLoading} error={previewError} range={currentRoleRange} onRangeChange={(range) => setDraft((current) => withRoleRange(current, previewRole, range))} events={previewRole === 'evaluation' ? displayedEvents : []} disabled={readonly || currentRoleRunId == null} title={`${previewRole[0].toUpperCase()}${previewRole.slice(1)} score timeline`} />
            {sameRunOverlap.length > 0 && <Alert color="red">Overlapping ranges on the same inference run: {sameRunOverlap.join(', ')}. Role ranges are inclusive, so one range ending exactly where another starts still shares a frame.</Alert>}
            <Text size="xs" c="dimmed">Training-data leakage, stored score completeness and compatibility are checked again by the server for every calculation.</Text>
          </StepCard>

          <StepCard index={5} title="Profile and independent calculations" subtitle="A, B and C can be calculated in any order. Only inputs used by a stage make that stage stale." color={STEP_COLORS[4]} complete={Boolean(projectedEvaluation && !draftProjection.dirty && canFinalizeEvaluation(projectedEvaluation))} action={!readonly && <Button size="compact-sm" variant="light" onClick={() => setProfileManagerOpen(true)}>Manage profiles</Button>}>
            {readonly ? <TextInput label="Evaluation profile" description="Frozen values from the finalized snapshot are shown below." value={selectedProfile?.name ?? `Profile #${selectedEvaluation?.profile_id ?? 'snapshot'}`} disabled /> : <Select label="Evaluation profile" description="Profiles are reusable; finalization stores a snapshot." data={profiles.map((profile) => ({ value: String(profile.id), label: profile.name }))} value={draft.profile_id == null ? null : String(draft.profile_id)} onChange={(value) => setDraft((current) => ({ ...current, profile_id: value ? Number(value) : null }))} searchable clearable />}
            {displayedProfile && <SimpleGrid cols={{ base: 2, md: 5 }}><Paper withBorder p="xs"><Text size="xs" c="dimmed">Normal window</Text><Text fw={600}>{displayedProfile.normal_window_duration_seconds}s</Text></Paper><Paper withBorder p="xs"><Text size="xs" c="dimmed">Normal buffer</Text><Text fw={600}>{displayedProfile.normal_window_buffer_seconds}s</Text></Paper><Paper withBorder p="xs"><Text size="xs" c="dimmed">Drift L</Text><Text fw={600}>{displayedProfile.drift_window_seconds}s</Text></Paper><Paper withBorder p="xs"><Text size="xs" c="dimmed">T₀</Text><Text fw={600}>{displayedProfile.false_alarm_horizon_seconds}s</Text></Paper><Paper withBorder p="xs"><Text size="xs" c="dimmed">Anticipation</Text><Text fw={600}>{displayedProfile.anticipation_seconds}s</Text></Paper></SimpleGrid>}
            {!readonly && selectedProfile && <><Button variant="subtle" color="gray" size="compact-sm" rightSection={profileOverridesOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />} onClick={() => setProfileOverridesOpen((current) => !current)}>Local profile overrides ({Object.keys(draft.profile_overrides).length})</Button><Collapse in={profileOverridesOpen}><Stack gap="xs"><Text size="xs" c="dimmed">Leave a field empty to inherit the selected profile. Each override only makes its dependent calculation stage stale when the draft is saved.</Text><SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}><NumberInput label="A · Normal window L_N (s)" min={0.001} placeholder={String(selectedProfile.normal_window_duration_seconds)} value={draft.profile_overrides.normal_window_duration_seconds ?? ''} onChange={(value) => setProfileOverride('normal_window_duration_seconds', value)} /><NumberInput label="A · Normal buffer P (s)" min={0} placeholder={String(selectedProfile.normal_window_buffer_seconds)} value={draft.profile_overrides.normal_window_buffer_seconds ?? ''} onChange={(value) => setProfileOverride('normal_window_buffer_seconds', value)} /><NumberInput label="B · Drift window L (s)" min={0.001} placeholder={String(selectedProfile.drift_window_seconds)} value={draft.profile_overrides.drift_window_seconds ?? ''} onChange={(value) => setProfileOverride('drift_window_seconds', value)} /><NumberInput label="C · False-alarm T₀ (s)" min={0.001} placeholder={String(selectedProfile.false_alarm_horizon_seconds)} value={draft.profile_overrides.false_alarm_horizon_seconds ?? ''} onChange={(value) => setProfileOverride('false_alarm_horizon_seconds', value)} /><NumberInput label="C · Anticipation (s)" min={0} placeholder={String(selectedProfile.anticipation_seconds)} value={draft.profile_overrides.anticipation_seconds ?? ''} onChange={(value) => setProfileOverride('anticipation_seconds', value)} /><NumberInput label="A/B · Epsilon" min={Number.MIN_VALUE} placeholder={String(selectedProfile.epsilon)} value={draft.profile_overrides.epsilon ?? ''} onChange={(value) => setProfileOverride('epsilon', value)} /></SimpleGrid></Stack></Collapse></>}
            {!readonly && targetEvents.length > 0 && <><Button variant="subtle" color="gray" size="compact-sm" rightSection={overridesOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />} onClick={() => setOverridesOpen((current) => !current)}>Per-event normal-window overrides ({Object.keys(draft.normal_window_overrides).length})</Button><Collapse in={overridesOpen}><Stack gap="xs"><Select label="Target event" data={targetEvents.map((event) => ({ value: event.event_id, label: `${event.name || event.event_id} · ${event.start_timestamp.replace('T', ' ')}` }))} value={overrideEventId} onChange={selectOverrideEvent} searchable clearable /><EvaluationTimeline preview={previews.evaluation ?? null} loading={false} range={overrideRange} onRangeChange={setOverrideRange} events={displayedEvents} disabled={!overrideEventId} title="Select this event’s normal window" selectionActionLabel="Use selection as normal window" /><Group justify="flex-end"><Button variant="default" size="compact-sm" disabled={!overrideEventId || !draft.normal_window_overrides[overrideEventId]} onClick={() => { if (!overrideEventId) return; setDraft((current) => { const next = { ...current.normal_window_overrides }; delete next[overrideEventId]; return { ...current, normal_window_overrides: next }; }); }}>Remove override</Button><Button size="compact-sm" disabled={!overrideEventId || !overrideRange.start_timestamp || !overrideRange.end_timestamp || overrideRange.start_timestamp >= overrideRange.end_timestamp} onClick={() => { if (!overrideEventId) return; setDraft((current) => ({ ...current, normal_window_overrides: { ...current.normal_window_overrides, [overrideEventId]: overrideRange } })); }}>Apply override</Button></Group></Stack></Collapse></>}
            <SimpleGrid cols={{ base: 1, lg: 3 }}>
              <StageAction stage="separation" label="A · Event separation" status={projectedEvaluation?.separation_status ?? 'not_calculated'} error={projectedEvaluation?.separation_error} disabled={readonly || !aReady} calculating={calculating} onCalculate={(stage) => void calculate(stage)} />
              <StageAction stage="drift" label="B · Score stability" status={projectedEvaluation?.drift_status ?? 'not_calculated'} error={projectedEvaluation?.drift_error} disabled={readonly || !bReady} calculating={calculating} onCalculate={(stage) => void calculate(stage)} />
              <StageAction stage="detection" label="C · Detection performance" status={projectedEvaluation?.detection_status ?? 'not_calculated'} error={projectedEvaluation?.detection_error} disabled={readonly || !cReady} calculating={calculating} onCalculate={(stage) => void calculate(stage)} />
            </SimpleGrid>
          </StepCard>

          <StepCard index={6} title="Metrics, diagnostics and finalization" subtitle="Review all eight principal metrics. Detection cards follow the active fixed quantile." color={STEP_COLORS[5]} complete={selectedEvaluation?.status === 'finalized'} action={selectedEvaluation?.status === 'draft' && <Button color="green" leftSection={<FileCheck2 size={15} />} disabled={draftProjection.dirty || !projectedEvaluation || !canFinalizeEvaluation(projectedEvaluation)} loading={saving} onClick={() => void finalize()}>Finalize snapshot</Button>}>
            {projectedEvaluation ? <EvaluationResults evaluation={projectedEvaluation} preview={previews.evaluation ?? null} events={displayedEvents} activeQuantile={draft.active_quantile} onActiveQuantileChange={(quantile) => setDraft((current) => ({ ...current, active_quantile: quantile }))} /> : <Alert color="blue">Save the draft, then calculate A, B and C independently to populate the eight metric cards and diagnostic plots.</Alert>}
          </StepCard>
        </>
      )}

      <EvaluationProfileManager opened={profileManagerOpen} onClose={() => setProfileManagerOpen(false)} profiles={profiles} onReload={async () => { await Promise.all([refreshProfiles(), refreshEvaluations()]); if (selectedEvaluation) replaceEvaluation(await getModelEvaluation(selectedEvaluation.id), false); }} onSelect={isEditorOpen && !readonly ? (profile) => { setDraft((current) => ({ ...current, profile_id: profile.id })); setProfileManagerOpen(false); } : undefined} />
      <EvaluationLabelSetManager opened={labelManagerOpen} onClose={() => setLabelManagerOpen(false)} labelSets={labelSets} datasets={datasets} initialLabelSetId={draft.label_set_id} initialDatasetId={evaluationRun?.training_dataset_id ?? null} suggestedEvent={suggestedEvent} onSuggestedEventConsumed={() => setSuggestedEvent(null)} onReload={async () => { await Promise.all([refreshLabelSets(), refreshEvaluations()]); if (selectedEvaluation) replaceEvaluation(await getModelEvaluation(selectedEvaluation.id), false); }} onSelect={isEditorOpen && !readonly ? (labelSet) => { setDraft((current) => ({ ...current, label_set_id: labelSet.id, selected_categories: labelSet.categories, normal_window_overrides: {} })); setLabelManagerOpen(false); } : undefined} />
    </Stack>
  );
}
