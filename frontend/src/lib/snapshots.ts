function normalizeForSnapshot(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeForSnapshot(item));
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entryValue]) => entryValue !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    return Object.fromEntries(entries.map(([key, entryValue]) => [key, normalizeForSnapshot(entryValue)]));
  }
  return value;
}

// Stable snapshots let builders compare loaded objects with editable drafts
// without depending on object insertion order from API responses or React state.
export function stableSnapshot(value: unknown): string {
  return JSON.stringify(normalizeForSnapshot(value));
}

export function snapshotsEqual(left: unknown, right: unknown): boolean {
  return stableSnapshot(left) === stableSnapshot(right);
}
