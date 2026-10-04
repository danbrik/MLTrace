import { Alert, Button, Group, Loader, Modal, Stack, Text } from '@mantine/core';
import { useEffect, useState } from 'react';
import { getTrainingLossPlot } from '../api';
import type { TrainingLossPlot, TrainingRun } from '../types';
import { downloadLossCsv, lossPlotRevision } from './lossPlot';

export function TrainingLossModal({ run, projectId, onClose }: {
  run: TrainingRun; projectId?: string; onClose: () => void;
}) {
  const revision = lossPlotRevision(run);
  const identity = `${projectId ?? 'current'}:${run.id}:${revision}`;
  const [result, setResult] = useState<{ identity: string; data: TrainingLossPlot } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let current = true;
    const controller = new AbortController();
    setLoading(true); setError(null);
    const timer = window.setTimeout(() => {
      getTrainingLossPlot(run.id, projectId, controller.signal).then(data => {
        if (current) setResult({ identity, data });
      }).catch(err => {
        if (current) setError(err instanceof Error ? err.message : 'Loss-Plot konnte nicht geladen werden.');
      }).finally(() => { if (current) setLoading(false); });
    }, 150);
    return () => { current = false; window.clearTimeout(timer); controller.abort(); };
  }, [run.id, projectId, identity, retry]);
  // Never expose a snapshot from before a restart or another run, even for one render.
  const snapshot = result?.identity === identity ? result.data : null;
  const available = Boolean(snapshot?.image_data_url) && !loading;
  return <Modal opened onClose={onClose} title="Loss-Plot" size="min(1100px, 95vw)">
    <Stack>
      <Text fw={600}>{run.training_pipeline_name}</Text>
      <Text size="sm" c="dimmed">Lauf {run.id} · {run.method_type} · {run.status}</Text>
      {loading && <Group><Loader size="sm" /><Text size="sm">Loss-Plot wird aktualisiert …</Text></Group>}
      {error && <Alert color="red" title="Loss-Plot konnte nicht geladen werden"><Text>{error}</Text><Button mt="sm" variant="light" onClick={() => setRetry(n => n + 1)}>Erneut versuchen</Button></Alert>}
      {snapshot && !snapshot.image_data_url && <Alert color="blue">Noch keine Loss-Werte vorhanden</Alert>}
      {snapshot?.image_data_url && <img src={snapshot.image_data_url} alt="Train Loss und Validation Loss nach Epoche" style={{ width: '100%', height: 'auto' }} />}
      {snapshot && !snapshot.has_validation && <Text size="sm" c="dimmed">Keine Validierung: Es wird ausschließlich Train Loss angezeigt.</Text>}
      <Group justify="flex-end">
        <Button component="a" href={available ? snapshot!.image_data_url! : undefined} download={`training-run-${run.id}-loss.png`} disabled={!available}>PNG herunterladen</Button>
        <Button disabled={!available} onClick={() => snapshot && downloadLossCsv(run.id, snapshot.metrics)}>CSV herunterladen</Button>
      </Group>
    </Stack>
  </Modal>;
}
