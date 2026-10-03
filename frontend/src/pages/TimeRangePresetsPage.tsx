import { Stack, Title } from '@mantine/core';
import { TimeRangePresetManager } from '../timeRangePresets/TimeRangePresetManager';

export function TimeRangePresetsPage({ projectId, active }: { projectId: string; active: boolean }) {
  return <Stack><Title order={2}>Gespeicherte Zeiträume</Title>
    <TimeRangePresetManager projectId={projectId} active={active} />
  </Stack>;
}
