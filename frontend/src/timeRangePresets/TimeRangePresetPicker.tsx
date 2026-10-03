import { Alert, Button, Group, Modal, Select, Stack } from '@mantine/core';
import { useState } from 'react';
import { formatPresetTime, rangeProblem } from './helpers';
import type { TimeRangeValue } from './types';
import { timeRangePresetStore, useTimeRangePresets } from './useTimeRangePresets';
import { TimeRangePresetActions, type PresetAction } from './TimeRangePresetActions';
import { TimeRangePresetManager } from './TimeRangePresetManager';

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
  const [action, setAction] = useState<PresetAction | null>(null);
  const [busy, setBusy] = useState(false);
  return <Stack gap="xs">
    <Select label="Gespeicherte Zeiträume" placeholder="Nach Namen suchen und übernehmen" searchable value={null}
      disabled={disabled || !!applyDisabledReason || loading}
      description={applyDisabledReason ?? 'Übernimmt ausschließlich Beginn und Ende (einschließlich).'}
      nothingFoundMessage="Keine gespeicherten Zeiträume gefunden"
      data={items.map(item => {
        const reason = rangeProblem(item, min, max);
        return { value: String(item.id), label: `${item.name} · ${formatPresetTime(item.start)} – ${formatPresetTime(item.end)}${reason ? ` · ${reason}` : ''}`, disabled: !!reason };
      })}
      onChange={id => {
        const selected = items.find(item => item.id === Number(id));
        if (selected && !disabled && !applyDisabledReason && !rangeProblem(selected, min, max)) onApply({ start: selected.start, end: selected.end });
      }} />
    <Group gap="xs">
      <Button size="xs" variant="light" disabled={disabled || !!rangeProblem(value)} onClick={() => {
        setAction({ mode: 'create', value: { name: '', start: value.start, end: value.end } }); setDialog('save');
      }}>Zeitraum speichern</Button>
      <Button size="xs" variant="subtle" disabled={disabled} onClick={() => setDialog('manage')}>Zeiträume bearbeiten / löschen</Button>
    </Group>
    {error && !dialog && <Alert color="red">{error}<Button variant="subtle" onClick={() => void timeRangePresetStore.load(projectId)}>Erneut laden</Button></Alert>}
    <Modal opened={dialog !== null} onClose={() => { if (!busy) setDialog(null); }}
      title={dialog === 'save' ? 'Zeitraum speichern' : 'Zeiträume bearbeiten / löschen'} size={dialog === 'save' ? 'lg' : '80%'}
      closeOnClickOutside={!busy} closeOnEscape={!busy} withCloseButton={!busy}>
      {dialog === 'save' && action && <TimeRangePresetActions projectId={projectId} action={action} onClose={() => setDialog(null)} onBusyChange={setBusy} />}
      {dialog === 'manage' && <TimeRangePresetManager projectId={projectId} onBusyChange={setBusy} />}
    </Modal>
  </Stack>;
}
