import {
  Alert,
  Button,
  FileInput,
  Group,
  Modal,
  NumberInput,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Download, Pencil, Plus, Save, Trash2, Upload } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  createEvaluationLabelSet,
  createEvaluationProfile,
  deleteEvaluationLabelSet,
  deleteEvaluationProfile,
  evaluationLabelSetCsvUrl,
  importEvaluationLabelCsv,
  previewEvaluationLabelCsv,
  updateEvaluationLabelSet,
  updateEvaluationProfile,
} from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import type {
  EvaluationLabelCsvPreview,
  EvaluationLabelEvent,
  EvaluationLabelSet,
  EvaluationLabelSetPayload,
  EvaluationProfile,
  EvaluationProfilePayload,
  TrainingDataset,
} from '../types';

function notifyError(title: string, error: unknown) {
  notifications.show({ color: 'red', title, message: error instanceof Error ? error.message : 'Unknown error' });
}

function numberValue(value: string | number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function defaultEvaluationProfilePayload(): EvaluationProfilePayload {
  return {
    name: '',
    description: '',
    normal_window_duration_seconds: 3600,
    normal_window_buffer_seconds: 0,
    drift_window_seconds: 3600,
    false_alarm_horizon_seconds: 3600,
    anticipation_seconds: 0,
    epsilon: 1e-12,
  };
}

function payloadFromProfile(profile: EvaluationProfile): EvaluationProfilePayload {
  return {
    name: profile.name,
    description: profile.description ?? '',
    normal_window_duration_seconds: profile.normal_window_duration_seconds,
    normal_window_buffer_seconds: profile.normal_window_buffer_seconds,
    drift_window_seconds: profile.drift_window_seconds,
    false_alarm_horizon_seconds: profile.false_alarm_horizon_seconds,
    anticipation_seconds: profile.anticipation_seconds,
    epsilon: profile.epsilon,
  };
}

export function EvaluationProfileManager({
  opened,
  onClose,
  profiles,
  onReload,
  onSelect,
}: {
  opened: boolean;
  onClose: () => void;
  profiles: EvaluationProfile[];
  onReload: () => Promise<void>;
  onSelect?: (profile: EvaluationProfile) => void;
}) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState<EvaluationProfilePayload>(defaultEvaluationProfilePayload);
  const [saving, setSaving] = useState(false);

  function edit(profile: EvaluationProfile) {
    setEditingId(profile.id);
    setForm(payloadFromProfile(profile));
  }

  function clear() {
    setEditingId(null);
    setForm(defaultEvaluationProfilePayload());
  }

  async function save() {
    if (!form.name.trim()) return;
    setSaving(true);
    try {
      const saved = editingId == null
        ? await createEvaluationProfile({ ...form, name: form.name.trim() })
        : await updateEvaluationProfile(editingId, { ...form, name: form.name.trim() });
      await onReload();
      onSelect?.(saved);
      clear();
      notifications.show({ color: 'green', title: 'Profile saved', message: saved.name });
    } catch (error) {
      notifyError('Could not save profile', error);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal opened={opened} onClose={onClose} title="Evaluation profiles" size="xl">
      <Stack>
        <Alert color="blue">Profiles define the normal window, fixed drift window, false-alarm horizon, anticipation window and numerical epsilon. Evaluations snapshot these values when finalized.</Alert>
        <ScrollArea h={220}>
          <Table striped highlightOnHover>
            <Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Normal window</Table.Th><Table.Th>Drift window</Table.Th><Table.Th>T₀</Table.Th><Table.Th /></Table.Tr></Table.Thead>
            <Table.Tbody>
              {profiles.map((profile) => (
                <Table.Tr key={profile.id}>
                  <Table.Td>{profile.name}</Table.Td>
                  <Table.Td>{profile.normal_window_duration_seconds}s + {profile.normal_window_buffer_seconds}s buffer</Table.Td>
                  <Table.Td>{profile.drift_window_seconds}s</Table.Td>
                  <Table.Td>{profile.false_alarm_horizon_seconds}s</Table.Td>
                  <Table.Td>
                    <Group justify="flex-end" gap="xs">
                      {onSelect && <Button size="compact-xs" variant="light" onClick={() => onSelect(profile)}>Use</Button>}
                      <Button size="compact-xs" variant="subtle" leftSection={<Pencil size={12} />} onClick={() => edit(profile)}>Edit</Button>
                      <Button
                        size="compact-xs"
                        variant="subtle"
                        color="red"
                        leftSection={<Trash2 size={12} />}
                        disabled={profile.is_update_locked}
                        onClick={async () => {
                          if (!window.confirm(`Delete profile “${profile.name}”?`)) return;
                          try { await deleteEvaluationProfile(profile.id); await onReload(); } catch (error) { notifyError('Could not delete profile', error); }
                        }}
                      >Delete</Button>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        <Text fw={600}>{editingId == null ? 'Create profile' : 'Edit profile'}</Text>
        <TextInput label="Name" value={form.name} onChange={(event) => setForm((current) => ({ ...current, name: event.currentTarget.value }))} required />
        <Textarea label="Description" value={form.description ?? ''} onChange={(event) => setForm((current) => ({ ...current, description: event.currentTarget.value }))} autosize minRows={2} />
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
          <NumberInput label="Normal window (seconds)" min={0.001} value={form.normal_window_duration_seconds} onChange={(value) => setForm((current) => ({ ...current, normal_window_duration_seconds: numberValue(value, 3600) }))} />
          <NumberInput label="Normal buffer (seconds)" min={0} value={form.normal_window_buffer_seconds} onChange={(value) => setForm((current) => ({ ...current, normal_window_buffer_seconds: numberValue(value, 0) }))} />
          <NumberInput label="Drift window L (seconds)" min={0.001} value={form.drift_window_seconds} onChange={(value) => setForm((current) => ({ ...current, drift_window_seconds: numberValue(value, 3600) }))} />
          <NumberInput label="False-alarm horizon T₀ (seconds)" min={0.001} value={form.false_alarm_horizon_seconds} onChange={(value) => setForm((current) => ({ ...current, false_alarm_horizon_seconds: numberValue(value, 3600) }))} />
          <NumberInput label="Anticipation (seconds)" min={0} value={form.anticipation_seconds} onChange={(value) => setForm((current) => ({ ...current, anticipation_seconds: numberValue(value, 0) }))} />
          <NumberInput label="Epsilon" min={0} value={form.epsilon} onChange={(value) => setForm((current) => ({ ...current, epsilon: numberValue(value, 1e-12) }))} />
        </SimpleGrid>
        <Group justify="flex-end">
          {editingId != null && <Button variant="default" onClick={clear}>Cancel edit</Button>}
          <Button leftSection={<Save size={15} />} loading={saving} disabled={!form.name.trim()} onClick={() => void save()}>Save profile</Button>
        </Group>
      </Stack>
    </Modal>
  );
}

function newEvent(type: 'target' | 'exclusion' = 'target'): EvaluationLabelEvent {
  return {
    event_id: crypto.randomUUID(),
    type,
    name: '',
    category: '',
    start_timestamp: '',
    end_timestamp: '',
    notes: '',
  };
}

export function EvaluationLabelSetManager({
  opened,
  onClose,
  labelSets,
  datasets,
  initialLabelSetId,
  initialDatasetId,
  suggestedEvent,
  onSuggestedEventConsumed,
  onReload,
  onSelect,
}: {
  opened: boolean;
  onClose: () => void;
  labelSets: EvaluationLabelSet[];
  datasets: TrainingDataset[];
  initialLabelSetId?: number | null;
  initialDatasetId?: number | null;
  suggestedEvent?: EvaluationLabelEvent | null;
  onSuggestedEventConsumed?: () => void;
  onReload: () => Promise<void>;
  onSelect?: (labelSet: EvaluationLabelSet) => void;
}) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [name, setName] = useState('');
  const [datasetId, setDatasetId] = useState<number | null>(initialDatasetId ?? null);
  const [events, setEvents] = useState<EvaluationLabelEvent[]>([]);
  const [editingEventIndex, setEditingEventIndex] = useState<number | null>(null);
  const [eventForm, setEventForm] = useState<EvaluationLabelEvent>(newEvent());
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvPreview, setCsvPreview] = useState<EvaluationLabelCsvPreview | null>(null);
  const [csvMode, setCsvMode] = useState<'replace' | 'append'>('replace');
  const [saving, setSaving] = useState(false);

  const compatibleLabelSets = useMemo(
    () => datasetId == null ? labelSets : labelSets.filter((set) => set.training_dataset_id === datasetId),
    [datasetId, labelSets],
  );
  const targetMetadataMissing = eventForm.type === 'target'
    && (!(eventForm.name ?? '').trim() || !(eventForm.category ?? '').trim());

  useEffect(() => {
    if (!opened) return;
    const selected = labelSets.find((set) => set.id === initialLabelSetId);
    if (selected) {
      setEditingId(selected.id);
      setName(selected.name);
      setDatasetId(selected.training_dataset_id);
      setEvents(selected.events ?? []);
    } else if (editingId == null) {
      setDatasetId(initialDatasetId ?? null);
    }
  }, [initialDatasetId, initialLabelSetId, labelSets, opened]);

  useEffect(() => {
    if (!opened || !suggestedEvent) return;
    setEventForm(suggestedEvent);
    setEditingEventIndex(null);
    onSuggestedEventConsumed?.();
  }, [onSuggestedEventConsumed, opened, suggestedEvent]);

  function clearSet() {
    setEditingId(null);
    setName('');
    setDatasetId(initialDatasetId ?? null);
    setEvents([]);
    setEventForm(newEvent());
    setEditingEventIndex(null);
    setCsvFile(null);
    setCsvPreview(null);
  }

  function editSet(labelSet: EvaluationLabelSet) {
    setEditingId(labelSet.id);
    setName(labelSet.name);
    setDatasetId(labelSet.training_dataset_id);
    setEvents(labelSet.events ?? []);
    setEventForm(newEvent());
    setEditingEventIndex(null);
    setCsvPreview(null);
  }

  function commitEvent() {
    if (targetMetadataMissing || !eventForm.start_timestamp || !eventForm.end_timestamp || eventForm.start_timestamp >= eventForm.end_timestamp) return;
    setEvents((current) => editingEventIndex == null
      ? [...current, eventForm]
      : current.map((event, index) => index === editingEventIndex ? eventForm : event));
    setEventForm(newEvent(eventForm.type));
    setEditingEventIndex(null);
  }

  async function previewCsv(file: File | null) {
    setCsvFile(file);
    setCsvPreview(null);
    if (!file || datasetId == null) return;
    try {
      const csvText = await file.text();
      setCsvPreview(await previewEvaluationLabelCsv({ csv_text: csvText, training_dataset_id: datasetId }));
    } catch (error) {
      notifyError('CSV validation failed', error);
    }
  }

  async function saveSet() {
    if (!name.trim() || datasetId == null) return;
    const payload: EvaluationLabelSetPayload = { name: name.trim(), training_dataset_id: datasetId, events };
    setSaving(true);
    try {
      const saved = editingId == null
        ? await createEvaluationLabelSet(payload)
        : await updateEvaluationLabelSet(editingId, payload);
      await onReload();
      onSelect?.(saved);
      notifications.show({ color: 'green', title: 'Ground truth saved', message: `${saved.name} · version ${saved.version}` });
      clearSet();
    } catch (error) {
      notifyError('Could not save ground truth', error);
    } finally {
      setSaving(false);
    }
  }

  async function importCsvToSavedSet() {
    if (editingId == null || datasetId == null || !csvFile || !csvPreview?.valid) return;
    setSaving(true);
    try {
      const imported = await importEvaluationLabelCsv(editingId, {
        training_dataset_id: datasetId,
        csv_text: await csvFile.text(),
        mode: csvMode,
      });
      await onReload();
      editSet(imported);
      onSelect?.(imported);
      notifications.show({ color: 'green', title: 'CSV imported', message: `${imported.events.length} intervals · version ${imported.version}` });
    } catch (error) {
      notifyError('Could not import CSV', error);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal opened={opened} onClose={onClose} title="Ground-truth label sets" size="90vw">
      <Stack>
        <Group justify="space-between">
          <Text fw={600}>Saved label sets</Text>
          <Button size="compact-sm" variant="light" leftSection={<Plus size={14} />} onClick={clearSet}>New label set</Button>
        </Group>
        <ScrollArea h={180}>
          <Table striped highlightOnHover>
            <Table.Thead><Table.Tr><Table.Th>Name</Table.Th><Table.Th>Dataset</Table.Th><Table.Th>Version</Table.Th><Table.Th>Targets</Table.Th><Table.Th>Exclusions</Table.Th><Table.Th /></Table.Tr></Table.Thead>
            <Table.Tbody>
              {compatibleLabelSets.map((labelSet) => (
                <Table.Tr key={labelSet.id}>
                  <Table.Td>{labelSet.name}</Table.Td>
                  <Table.Td>{labelSet.training_dataset_name ?? datasets.find((dataset) => dataset.id === labelSet.training_dataset_id)?.name ?? `#${labelSet.training_dataset_id}`}</Table.Td>
                  <Table.Td>{labelSet.version}</Table.Td>
                  <Table.Td>{labelSet.events.filter((event) => event.type === 'target').length}</Table.Td>
                  <Table.Td>{labelSet.events.filter((event) => event.type === 'exclusion').length}</Table.Td>
                  <Table.Td>
                    <Group justify="flex-end" gap="xs">
                      {onSelect && <Button size="compact-xs" variant="light" onClick={() => onSelect(labelSet)}>Use</Button>}
                      <Button component="a" href={evaluationLabelSetCsvUrl(labelSet.id)} download size="compact-xs" variant="subtle" leftSection={<Download size={12} />}>CSV</Button>
                      <Button size="compact-xs" variant="subtle" leftSection={<Pencil size={12} />} onClick={() => editSet(labelSet)}>Edit</Button>
                      <Button
                        size="compact-xs"
                        variant="subtle"
                        color="red"
                        disabled={labelSet.is_update_locked}
                        leftSection={<Trash2 size={12} />}
                        onClick={async () => {
                          if (!window.confirm(`Delete label set “${labelSet.name}”?`)) return;
                          try { await deleteEvaluationLabelSet(labelSet.id); await onReload(); } catch (error) { notifyError('Could not delete label set', error); }
                        }}
                      >Delete</Button>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>

        <Text fw={600}>{editingId == null ? 'Create label set' : 'Edit label set'}</Text>
        <SimpleGrid cols={{ base: 1, sm: 2 }}>
          <TextInput label="Name" required value={name} onChange={(event) => setName(event.currentTarget.value)} />
          <Select
            label="Inference dataset"
            required
            data={datasets.map((dataset) => ({ value: String(dataset.id), label: dataset.name }))}
            value={datasetId == null ? null : String(datasetId)}
            onChange={(value) => setDatasetId(value ? Number(value) : null)}
            searchable
            disabled={editingId != null}
          />
        </SimpleGrid>

        <SimpleGrid cols={{ base: 1, md: 2 }}>
          <Stack gap="xs">
            <Text fw={600} size="sm">Add or edit interval</Text>
            <SimpleGrid cols={{ base: 1, sm: 3 }}>
              <Select label="Type" data={[{ value: 'target', label: 'Target event' }, { value: 'exclusion', label: 'Exclusion / invalid' }]} value={eventForm.type} onChange={(value) => setEventForm((current) => ({ ...current, type: (value ?? 'target') as 'target' | 'exclusion' }))} />
              <TextInput label="Name" value={eventForm.name ?? ''} onChange={(event) => setEventForm((current) => ({ ...current, name: event.currentTarget.value }))} required={eventForm.type === 'target'} />
              <TextInput label="Category" value={eventForm.category ?? ''} onChange={(event) => setEventForm((current) => ({ ...current, category: event.currentTarget.value }))} required={eventForm.type === 'target'} />
            </SimpleGrid>
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <DateTime24Input label="Start" value={eventForm.start_timestamp} onChange={(value) => setEventForm((current) => ({ ...current, start_timestamp: value }))} />
              <DateTime24Input label="End" value={eventForm.end_timestamp} onChange={(value) => setEventForm((current) => ({ ...current, end_timestamp: value }))} />
            </SimpleGrid>
            <Textarea label="Notes" value={eventForm.notes ?? ''} onChange={(event) => setEventForm((current) => ({ ...current, notes: event.currentTarget.value }))} autosize minRows={2} />
            {targetMetadataMissing && <Text size="xs" c="red">Target events require both a name and category.</Text>}
            <Group justify="flex-end">
              {editingEventIndex != null && <Button variant="default" size="compact-sm" onClick={() => { setEditingEventIndex(null); setEventForm(newEvent()); }}>Cancel</Button>}
              <Button size="compact-sm" leftSection={<Plus size={14} />} disabled={targetMetadataMissing || !eventForm.start_timestamp || !eventForm.end_timestamp || eventForm.start_timestamp >= eventForm.end_timestamp} onClick={commitEvent}>{editingEventIndex == null ? 'Add interval' : 'Apply change'}</Button>
            </Group>
          </Stack>
          <Stack gap="xs">
            <Text fw={600} size="sm">CSV import with validation preview</Text>
            <FileInput accept=".csv,text/csv" value={csvFile} onChange={(file) => void previewCsv(file)} leftSection={<Upload size={15} />} placeholder={datasetId == null ? 'Select a dataset first' : 'Choose CSV'} disabled={datasetId == null} />
            {editingId != null && <Select label="Import mode" data={[{ value: 'replace', label: 'Replace all intervals' }, { value: 'append', label: 'Append to current intervals' }]} value={csvMode} onChange={(value) => setCsvMode((value ?? 'replace') as 'replace' | 'append')} />}
            <Text size="xs" c="dimmed">Columns: event_id, type, name, category, start_timestamp, end_timestamp, notes.</Text>
            {csvPreview && (
              <Alert color={csvPreview.valid ? 'green' : 'red'}>
                <Stack gap={4}>
                <Text size="sm">{csvPreview.events.length} valid rows · {csvPreview.errors.length} errors</Text>
                  {csvPreview.errors.map((item, index) => <Text key={`e-${index}`} size="xs">Row {item.row}: {item.message}</Text>)}
                  <Group gap="xs">
                    <Button size="compact-xs" variant="light" disabled={!csvPreview.valid} onClick={() => setEvents(csvPreview.events)}>Use rows in editor</Button>
                    {editingId != null && <Button size="compact-xs" disabled={!csvPreview.valid} loading={saving} onClick={() => void importCsvToSavedSet()}>Import into saved set</Button>}
                  </Group>
                </Stack>
              </Alert>
            )}
          </Stack>
        </SimpleGrid>

        <ScrollArea h={250}>
          <Table striped highlightOnHover miw={900}>
            <Table.Thead><Table.Tr><Table.Th>Type</Table.Th><Table.Th>Name</Table.Th><Table.Th>Category</Table.Th><Table.Th>Start</Table.Th><Table.Th>End</Table.Th><Table.Th>Notes</Table.Th><Table.Th /></Table.Tr></Table.Thead>
            <Table.Tbody>
              {events.map((event, index) => (
                <Table.Tr key={event.event_id}>
                  <Table.Td>{event.type}</Table.Td><Table.Td>{event.name || '—'}</Table.Td><Table.Td>{event.category || '—'}</Table.Td>
                  <Table.Td>{event.start_timestamp.replace('T', ' ')}</Table.Td><Table.Td>{event.end_timestamp.replace('T', ' ')}</Table.Td><Table.Td>{event.notes || '—'}</Table.Td>
                  <Table.Td><Group gap="xs" justify="flex-end"><Button size="compact-xs" variant="subtle" onClick={() => { setEditingEventIndex(index); setEventForm(event); }}>Edit</Button><Button size="compact-xs" variant="subtle" color="red" onClick={() => setEvents((current) => current.filter((_, candidate) => candidate !== index))}>Remove</Button></Group></Table.Td>
                </Table.Tr>
              ))}
              {events.length === 0 && <Table.Tr><Table.Td colSpan={7}><Text c="dimmed" ta="center">No intervals yet. Add them here, import CSV, or drag them in the evaluation timeline.</Text></Table.Td></Table.Tr>}
            </Table.Tbody>
          </Table>
        </ScrollArea>
        <Group justify="space-between">
          <Text size="xs" c="dimmed">Overlapping target events and out-of-dataset intervals are rejected by the server when saved.</Text>
          <Group>
            {editingId != null && <Button variant="default" onClick={clearSet}>Cancel edit</Button>}
            <Button leftSection={<Save size={15} />} loading={saving} disabled={!name.trim() || datasetId == null} onClick={() => void saveSet()}>Save label set</Button>
          </Group>
        </Group>
      </Stack>
    </Modal>
  );
}
