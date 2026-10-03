import { Alert, Button, Group, Stack, Table, Text, TextInput } from '@mantine/core';
import { useState } from 'react';
import { DateTime24Input } from '../components/DateTime24Input';
import { formatPresetTime, formatRangeDuration, rangeProblem } from './helpers';
import type { TimeRangePreset, TimeRangePresetInput } from './types';
import { timeRangePresetStore } from './useTimeRangePresets';

export type PresetAction =
  | { mode: 'create'; value: TimeRangePresetInput }
  | { mode: 'edit' | 'details' | 'delete'; value: TimeRangePreset };

export function TimeRangePresetActions({ projectId, action, onClose, onBusyChange }: {
  projectId: string; action: PresetAction; onClose: () => void; onBusyChange?: (busy: boolean) => void;
}) {
  const [draft, setDraft] = useState<TimeRangePresetInput>({ name: action.value.name, start: action.value.start, end: action.value.end });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const problem = rangeProblem(draft);
  async function save() {
    setBusy(true); onBusyChange?.(true); setError(null);
    try {
      if (action.mode === 'delete') await timeRangePresetStore.remove(projectId, action.value.id);
      else if (action.mode === 'edit') await timeRangePresetStore.update(projectId, action.value.id, draft);
      else if (action.mode === 'create') await timeRangePresetStore.create(projectId, draft);
      onClose();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); onBusyChange?.(false); }
  }
  if (action.mode === 'details') {
    const row = action.value;
    return <Stack>
      <Text fw={600}>Eigenschaften von „{row.name}“</Text>
      <Table><Table.Tbody>{[
        ['ID', row.id], ['Name', row.name], ['Beginn (einschließlich)', formatPresetTime(row.start)],
        ['Ende (einschließlich)', formatPresetTime(row.end)], ['Dauer', formatRangeDuration(row)],
        ['Erstellt (UTC)', formatPresetTime(row.created_at)], ['Geändert (UTC)', formatPresetTime(row.updated_at)],
      ].map(([label, value]) => <Table.Tr key={label}><Table.Th>{label}</Table.Th><Table.Td>{value}</Table.Td></Table.Tr>)}</Table.Tbody></Table>
      <Button variant="default" w="fit-content" onClick={onClose}>Zurück zur Liste</Button>
    </Stack>;
  }
  return <Stack gap="sm">
    {error && <Alert color="red">{error}</Alert>}
    {action.mode === 'delete' ? <>
      <Text fw={600}>Gespeicherten Zeitraum „{action.value.name}“ löschen?</Text>
      <Text size="sm">Die Vorlage wird aus der gemeinsamen Liste dieses Projekts entfernt. Bereits übernommene Formularwerte und gespeicherte Analyse-Läufe bleiben erhalten.</Text>
    </> : <>
      <Text fw={600}>{action.mode === 'create' ? 'Neuer Zeitraum' : `Zeitraum „${action.value.name}“ bearbeiten`}</Text>
      <TextInput label="Name" value={draft.name} maxLength={255} required disabled={busy}
        onChange={event => setDraft({ ...draft, name: event.currentTarget.value })} />
      <DateTime24Input label="Gespeicherter Beginn (einschließlich)" value={draft.start} disabled={busy} onChange={start => setDraft({ ...draft, start })} />
      <DateTime24Input label="Gespeichertes Ende (einschließlich)" value={draft.end} disabled={busy} onChange={end => setDraft({ ...draft, end })} />
      <Text size="sm" c="dimmed">Beginn und Ende zählen mit; beide dürfen auf denselben Zeitpunkt fallen.</Text>
      {problem && <Text size="sm" c="red">{problem}</Text>}
    </>}
    <Group>
      <Button color={action.mode === 'delete' ? 'red' : undefined} loading={busy}
        disabled={action.mode !== 'delete' && (!!problem || !draft.name.trim())} onClick={() => void save()}>
        {action.mode === 'delete' ? 'Gespeicherten Zeitraum löschen' : 'Speichern'}
      </Button>
      <Button variant="default" disabled={busy} onClick={onClose}>Abbrechen</Button>
    </Group>
  </Stack>;
}
