import type { TimeRangePreset, TimeRangePresetInput } from './types';

type Client = {
  list: (project: string) => Promise<TimeRangePreset[]>;
  create: (project: string, input: TimeRangePresetInput) => Promise<TimeRangePreset>;
  update: (project: string, id: number, input: TimeRangePresetInput) => Promise<TimeRangePreset>;
  remove: (project: string, id: number) => Promise<void>;
};
type Snapshot = { items: TimeRangePreset[]; loading: boolean; error: string | null };
type Entry = { snapshot: Snapshot; listeners: Set<() => void>; version: number; pending?: Promise<void> };

export function createTimeRangePresetStore(client: Client) {
  const projects = new Map<string, Entry>();
  const entry = (project: string) => {
    if (!projects.has(project)) projects.set(project, { snapshot: { items: [], loading: false, error: null }, listeners: new Set(), version: 0 });
    return projects.get(project)!;
  };
  const emit = (item: Entry, snapshot: Snapshot) => {
    item.snapshot = snapshot;
    item.listeners.forEach(listener => listener());
  };
  const changed = (project: string, row: TimeRangePreset | null, id: number) => {
    const item = entry(project);
    ++item.version; // Ignore lists started before this committed mutation.
    const items = item.snapshot.items.filter(value => value.id !== id);
    if (row) items.push(row);
    items.sort((a, b) => a.name.localeCompare(b.name));
    emit(item, { items, loading: false, error: null });
  };
  return {
    snapshot: (project: string) => entry(project).snapshot,
    subscribe(project: string, listener: () => void) {
      const item = entry(project);
      item.listeners.add(listener);
      return () => { item.listeners.delete(listener); };
    },
    load(project: string): Promise<void> {
      const item = entry(project);
      if (item.pending) return item.pending;
      emit(item, { ...item.snapshot, loading: true, error: null });
      item.pending = (async () => {
        // A mutation during a list request requires a fresh list, not its stale response.
        for (;;) {
          const version = item.version;
          try {
            const items = await client.list(project);
            if (item.version !== version) continue;
            emit(item, { items, loading: false, error: null });
          } catch (error) {
            if (item.version !== version) continue;
            emit(item, { ...item.snapshot, loading: false, error: String(error instanceof Error ? error.message : error) });
          }
          break;
        }
      })().finally(() => { item.pending = undefined; });
      return item.pending;
    },
    async create(project: string, input: TimeRangePresetInput) {
      const row = await client.create(project, input);
      changed(project, row, row.id);
      return row;
    },
    async update(project: string, id: number, input: TimeRangePresetInput) {
      const row = await client.update(project, id, input);
      changed(project, row, id);
      return row;
    },
    async remove(project: string, id: number) {
      await client.remove(project, id);
      changed(project, null, id);
    },
  };
}
