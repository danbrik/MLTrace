import { Alert, Button, Group, Modal, Paper, Select, Stack, Text, TextInput } from '@mantine/core';
import { useState } from 'react';
import { DateTime24Input } from '../components/DateTime24Input';
import { formatPresetTime, rangeProblem } from './helpers';
import type { TimeRangePreset, TimeRangePresetInput, TimeRangeValue } from './types';
import { timeRangePresetStore, useTimeRangePresets } from './useTimeRangePresets';

export type TimeRangePresetPickerProps = {
  projectId: string;
  active?: boolean;
  value: TimeRangeValue;
  onApply: (range: TimeRangeValue) => void;
  min?: string;
  max?: string;
  disabled?: boolean;
  applyDisabledReason?: string;
};

/** Project-keyed boundary also resets open dialogs on a project switch. */
export function TimeRangePresetPicker(props: TimeRangePresetPickerProps) {
  return <Picker key={props.projectId} {...props} />;
}

function Picker({ projectId, active = true, value, onApply, min, max, disabled, applyDisabledReason }: TimeRangePresetPickerProps) {
  const { items, loading, error } = useTimeRangePresets(projectId, active);
  const [dialog, setDialog] = useState<'save' | 'manage' | null>(null);
  const [draft, setDraft] = useState<TimeRangePresetInput>({ name: '', ...value });
  const [editingId, setEditingId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState<TimeRangePreset | null>(null);
  const [search, setSearch] = useState('');
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const problem = rangeProblem(draft);
  const showTimes = (range: TimeRangeValue) => `${formatPresetTime(range.start)} – ${formatPresetTime(range.end)}`;
  function open(mode: 'save' | 'manage') {
    setDraft({ name: '', start: value.start, end: value.end });
    setEditingId(null); setDeleting(null); setFailure(null); setSearch(''); setDialog(mode);
    void timeRangePresetStore.load(projectId);
  }
  async function perform(work: () => Promise<unknown>, done: () => void) {
    setBusy(true); setFailure(null);
    try { await work(); done(); }
    catch (reason) { setFailure(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  }
  function submit() {
    if (problem || !draft.name.trim()) return;
    void perform(() => editingId === null
      ? timeRangePresetStore.create(projectId, draft)
      : timeRangePresetStore.update(projectId, editingId, draft),
    () => { if (dialog === 'save') setDialog(null); setEditingId(null); });
  }
  const loadError = error && <Alert color="red">{error}<Button variant="subtle" onClick={() => void timeRangePresetStore.load(projectId)}>Erneut laden</Button></Alert>;
  const editor = <Stack gap="sm">
    <TextInput label="Name" value={draft.name} maxLength={255} required disabled={busy}
      onChange={event => setDraft({ ...draft, name: event.currentTarget.value })} />
    <DateTime24Input label="Gespeicherter Beginn" value={draft.start} disabled={busy}
      onChange={start => setDraft({ ...draft, start })} />
    <DateTime24Input label="Gespeichertes Ende" value={draft.end} disabled={busy}
      onChange={end => setDraft({ ...draft, end })} />
    {problem && <Text size="sm" c="red">{problem}</Text>}
    <Group>
      <Button onClick={submit} loading={busy} disabled={!!problem || !draft.name.trim()}>Speichern</Button>
      <Button variant="default" disabled={busy} onClick={() => dialog === 'save' ? setDialog(null) : setEditingId(null)}>Abbrechen</Button>
    </Group>
  </Stack>;

  return <Stack gap="xs">
    <Select label="Gespeicherte Zeiträume" placeholder="Nach Namen suchen und übernehmen" searchable value={null}
      disabled={disabled || !!applyDisabledReason || loading}
      description={applyDisabledReason ?? 'Übernimmt ausschließlich Beginn und Ende.'}
      nothingFoundMessage="Keine gespeicherten Zeiträume gefunden"
      data={items.map(item => {
        const reason = rangeProblem(item, min, max);
        return { value: String(item.id), label: `${item.name} · ${showTimes(item)}${reason ? ` · ${reason}` : ''}`, disabled: !!reason };
      })}
      onChange={id => {
        const selected = items.find(item => item.id === Number(id));
        if (selected && !disabled && !applyDisabledReason && !rangeProblem(selected, min, max)) onApply({ start: selected.start, end: selected.end });
      }} />
    <Group gap="xs">
      <Button size="xs" variant="light" disabled={disabled || !!rangeProblem(value)} onClick={() => open('save')}>Zeitraum speichern</Button>
      <Button size="xs" variant="subtle" disabled={disabled} onClick={() => open('manage')}>Zeiträume verwalten</Button>
    </Group>
    {!dialog && loadError}
    <Modal opened={dialog !== null} onClose={() => { if (!busy) setDialog(null); }}
      title={dialog === 'save' ? 'Zeitraum speichern' : 'Zeiträume verwalten'} size="lg" closeOnClickOutside={!busy} closeOnEscape={!busy} withCloseButton={!busy}>
      <Stack>
        <Text size="sm" c="dimmed">Diese Liste gilt für das aktuelle Projekt. Änderungen wirken sich nicht auf übernommene Zeiten oder bestehende Analyse-Läufe aus.</Text>
        {loadError}
        {failure && <Alert color="red">{failure}</Alert>}
        {dialog === 'save' ? editor : <>
          <TextInput label="Zeiträume nach Namen suchen" value={search} onChange={event => setSearch(event.currentTarget.value)} />
          {editingId !== null && <Paper withBorder p="sm">{editor}</Paper>}
          {deleting && <Alert color="red" title={`„${deleting.name}“ löschen?`}>
            <Group mt="sm">
              <Button color="red" loading={busy} onClick={() => void perform(() => timeRangePresetStore.remove(projectId, deleting.id), () => { setDeleting(null); if (editingId === deleting.id) setEditingId(null); })}>Endgültig löschen</Button>
              <Button variant="default" disabled={busy} onClick={() => setDeleting(null)}>Abbrechen</Button>
            </Group>
          </Alert>}
          {loading && <Text size="sm">Zeiträume werden geladen …</Text>}
          {!loading && items.length === 0 && <Text c="dimmed">Noch keine Zeiträume gespeichert.</Text>}
          {items.filter(item => item.name.toLocaleLowerCase().includes(search.toLocaleLowerCase())).map(item => <Paper key={item.id} withBorder p="sm">
            <Text fw={600}>{item.name}</Text><Text size="sm">{showTimes(item)}</Text>
            <Group mt="xs">
              <Button size="xs" variant="light" disabled={busy} onClick={() => { setEditingId(item.id); setDraft({ name: item.name, start: item.start, end: item.end }); setDeleting(null); setFailure(null); }}>Bearbeiten</Button>
              <Button size="xs" color="red" variant="subtle" disabled={busy} onClick={() => { setDeleting(item); setFailure(null); }}>Löschen</Button>
            </Group>
          </Paper>)}
        </>}
      </Stack>
    </Modal>
  </Stack>;
}
