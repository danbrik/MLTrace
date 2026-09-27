import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { createTimeRangePreset, deleteTimeRangePreset, listTimeRangePresets, updateTimeRangePreset } from '../api';
import { createTimeRangePresetStore } from './store';

export const timeRangePresetStore = createTimeRangePresetStore({
  list: listTimeRangePresets, create: createTimeRangePreset, update: updateTimeRangePreset, remove: deleteTimeRangePreset,
});

export function useTimeRangePresets(projectId: string, active = true) {
  const subscribe = useCallback((listener: () => void) => timeRangePresetStore.subscribe(projectId, listener), [projectId]);
  const snapshot = useCallback(() => timeRangePresetStore.snapshot(projectId), [projectId]);
  const state = useSyncExternalStore(subscribe, snapshot);
  useEffect(() => { if (active) void timeRangePresetStore.load(projectId); }, [projectId, active]);
  return state;
}
