export type ResourceKey =
  | 'datasets'
  | 'trainingDatasets'
  | 'preprocessingPipelines'
  | 'preprocessingSteps'
  | 'methodDefinitions'
  | 'methodLayers'
  | 'methodConfigurations'
  | 'trainingPipelines'
  | 'trainingRuns'
  | 'testingRuns'
  | 'heatmaps'
  | 'heatmapRanges'
  | 'inspectRuns'
  | 'analysisLayouts'
  | 'optimizationStudies'
  | 'rois';

type CacheEntry<T> = {
  value?: T;
  expiresAt: number;
  revision?: string;
  inFlight?: Promise<T>;
};

const entries = new Map<ResourceKey, CacheEntry<unknown>>();

function nowMs(): number {
  return Date.now();
}

/**
 * Small first-party resource cache for page-level lists. It intentionally sits
 * below the existing API functions so pages can keep their current refresh
 * structure while repeated navigation reuses already loaded data.
 */
export async function cachedResource<T>(
  key: ResourceKey,
  loader: () => Promise<T>,
  options: {
    ttlMs: number;
    revision?: () => Promise<string | undefined>;
  },
): Promise<T> {
  const now = nowMs();
  const entry = (entries.get(key) as CacheEntry<T> | undefined) ?? { expiresAt: 0 };

  if (entry.value !== undefined && entry.expiresAt > now) {
    return entry.value;
  }

  if (entry.inFlight) {
    return entry.inFlight;
  }

  let latestRevision = entry.revision;
  if (entry.value !== undefined && options.revision) {
    const revision = await options.revision().catch(() => undefined);
    latestRevision = revision ?? latestRevision;
    if (revision && revision === entry.revision) {
      entry.expiresAt = now + options.ttlMs;
      entries.set(key, entry);
      return entry.value;
    }
  }

  const inFlight = loader()
    .then((value) => {
      entries.set(key, {
        value,
        expiresAt: nowMs() + options.ttlMs,
        revision: latestRevision,
      });
      return value;
    })
    .finally(() => {
      const current = entries.get(key);
      if (current?.inFlight === inFlight) {
        delete current.inFlight;
      }
    });

  entries.set(key, { ...entry, inFlight });
  return inFlight;
}

export function setResourceRevision(key: ResourceKey, revision: string | undefined): void {
  const entry = entries.get(key);
  if (!entry) return;
  entry.revision = revision;
}

export function invalidateResources(keys: ResourceKey[]): void {
  for (const key of keys) {
    entries.delete(key);
  }
}

export function invalidateAllResources(): void {
  entries.clear();
}
