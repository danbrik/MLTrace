import { useCallback, useRef, useState } from 'react';

export function usePendingIds() {
  const [pendingIds, setPendingIds] = useState<Set<string | number>>(() => new Set());
  const pendingRef = useRef<Set<string | number>>(new Set());

  const isPending = useCallback((id: string | number) => pendingIds.has(id), [pendingIds]);

  const runPending = useCallback(
    async <T,>(id: string | number, action: () => Promise<T>): Promise<T | undefined> => {
      if (pendingRef.current.has(id)) return undefined;
      pendingRef.current.add(id);
      setPendingIds((current) => new Set(current).add(id));
      try {
        return await action();
      } finally {
        pendingRef.current.delete(id);
        setPendingIds((current) => {
          const next = new Set(current);
          next.delete(id);
          return next;
        });
      }
    },
    [],
  );

  return { pendingIds, isPending, runPending };
}
