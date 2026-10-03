import { Alert, Button, Group, Image, Loader, Paper, Progress, Select, Slider, Stack, Table, Text, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { abortVarianceRoiJob, evaluateVarianceRoi, getVarianceRoiState, prepareVarianceRoi, varianceRoiArtifactUrl, varianceRoiImageUrl } from '../api';
import { RoiEditor } from './RoiEditor';
import { roiCoordinates } from './roiGeometry';
import { displayTime } from './helpers';
import type { RoiConfig, RoiState } from './roiTypes';
const fmt = (value: number | null) => value === null ? '—' : value.toLocaleString('de-DE', {maximumFractionDigits: 2});

export function VarianceRoiPanel({runId, projectId, active}: {runId: number; projectId: string; active: boolean}) {
  const [state, setState] = useState<RoiState | null>(null);
  const [config, setConfig] = useState<RoiConfig | null>(null);
  const [pair, setPair] = useState('0');
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const live = useRef(false), initialized = useRef(false), requestRevision = useRef(0);
  const basis = state?.basis;
  const saved = state?.saved;
  const result = saved?.result;
  const job = state?.job;
  const running = !!job && ['queued', 'running'].includes(job.status);
  async function refresh() {
    const revision = ++requestRevision.current;
    const next = await getVarianceRoiState(runId, projectId);
    if (live.current && revision === requestRevision.current) setState(next);
  }
  useEffect(() => {
    if (!active) return;
    live.current = true;
    setBusy(false);
    let pending = false;
    async function poll() {
      if (pending) return;
      pending = true;
      try { await refresh(); } catch (reason) {if (live.current) setError(String(reason));}
      finally {pending = false;}
    }
    void poll();
    const timer = window.setInterval(() => void poll(), 1500);
    return () => {live.current = false; requestRevision.current++; window.clearInterval(timer);};
  }, [runId, projectId, active]);
  useEffect(() => {
    if (!basis) return;
    if (!initialized.current) {
      initialized.current = true;
      setConfig(result ? {roi: result.roi, opacity: result.opacity} : {roi: {x: 0, y: 0, width: basis.width, height: basis.height}, opacity: .5});
      setEditing(!result);
    } else if (!editing && result) {
      setConfig({roi: result.roi, opacity: result.opacity});
    }
  }, [!!basis, saved?.id, editing]);
  async function action(work: () => Promise<unknown>, finish = false) {
    setBusy(true); setError(null);
    try {
      await work();
      if (!live.current) return;
      if (finish) setEditing(false);
      await refresh();
    } catch (reason) {if (live.current) setError(reason instanceof Error ? reason.message : String(reason));}
    finally {if (live.current) setBusy(false);}
  }
  return <Paper withBorder p="lg"><Stack>
    <Title order={4}>ROI-Auswertung · Lauf #{runId}</Title>
    <Text size="sm">Die Heatmap zeigt Anomalievarianz − Normalvarianz. Die Tabelle zählt ausschließlich die positive Varianzzunahme.</Text>
    {error && <Alert color="red" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
    {!state && !error && <Loader size="sm" />}
    {state && !state.ready && <>
      <Text>Für diesen gespeicherten Lauf fehlen noch die numerischen ROI-Daten. Die Nachbereitung verwendet die eingefrorene Bildauswahl und Pipeline; der ursprüngliche Vergleich bleibt erhalten.</Text>
      <Button w="fit-content" disabled={busy || running} onClick={() => void action(() => prepareVarianceRoi(runId, projectId))}>ROI-Daten nachberechnen</Button>
    </>}
    {running && <>
      <Text>{job!.status === 'queued' ? 'Wartet im Scheduler' : job!.operation === 'prepare' ? 'ROI-Daten werden nachberechnet' : 'ROI-Plot und Tabelle werden erstellt'} · {job!.processed_images}/{job!.total_images ?? '…'}</Text>
      <Progress animated value={job!.total_images ? 100 * job!.processed_images / job!.total_images : 0} />
      <Button color="orange" variant="light" w="fit-content" disabled={busy || job!.cancel_requested} onClick={() => void action(() => abortVarianceRoiJob(job!.id, projectId))}>{job!.cancel_requested ? 'Abbruch angefordert' : 'ROI-Auftrag abbrechen'}</Button>
    </>}
    {job?.error_message && <Alert color={job.status === 'aborted' ? 'yellow' : 'red'}>{job.error_message}</Alert>}
    {basis && config && <>
      <Text size="sm">{basis.dataset_name} · {basis.pipeline_name} · {basis.width} × {basis.height} Pixel</Text>
      {basis.pairs.map(item => <Text key={item.label} size="sm">{item.label}: Normalzustand {displayTime(item.periods.normal.start)} – {displayTime(item.periods.normal.end)} ({item.counts.normal} Bilder) · Anomaliephase {displayTime(item.periods.anomaly.start)} – {displayTime(item.periods.anomaly.end)} ({item.counts.anomaly} Bilder)</Text>)}
      {editing ? <>
        <Select label="Paar im Editor" value={pair} allowDeselect={false} disabled={busy || running} data={basis.pairs.map((item, i) => ({value: String(i), label: item.label}))} onChange={value => setPair(value ?? '0')} />
        <Text size="sm">Deckkraft der Heatmap: {Math.round(config.opacity * 100)} %</Text>
        <Slider thumbLabel="Deckkraft der Heatmap" value={config.opacity * 100} min={0} max={100} disabled={busy || running} onChange={value => setConfig({...config, opacity: value / 100})} />
        <RoiEditor key={pair} width={basis.width} height={basis.height} value={config.roi} opacity={config.opacity} disabled={busy || running}
          background={varianceRoiImageUrl(runId, Number(pair), 'background', projectId)} heatmap={varianceRoiImageUrl(runId, Number(pair), 'heatmap', projectId)}
          onChange={roi => setConfig({...config, roi})} />
        <Text size="sm">{roiCoordinates(config.roi)}</Text>
        <Group><Button disabled={busy || running} onClick={() => void action(() => evaluateVarianceRoi(runId, config, projectId), true)}>Fertig</Button>
          {result && <Button variant="subtle" disabled={busy || running} onClick={() => setEditing(false)}>Bearbeitung verwerfen</Button>}
        </Group>
      </> : <>
        <Button w="fit-content" variant="light" disabled={busy || running} onClick={() => setEditing(true)}>ROI bearbeiten</Button>
        {result && saved && <>
          {running && <Text size="sm" c="dimmed">Das bisher gespeicherte Ergebnis bleibt bis zum erfolgreichen Abschluss sichtbar.</Text>}
          <Text size="sm">{roiCoordinates(result.roi)} · Deckkraft: {Math.round(result.opacity * 100)} %</Text>
          {result.warnings.map(warning => <Alert key={warning} color="yellow">{warning}</Alert>)}
          {result.mean_increase_percent !== null ? <Text fw={600}>Die ROI umfasst {fmt(result.area_percent)} % der Bildfläche und enthält im Mittel {fmt(result.mean_increase_percent)} % der positiven Varianzzunahme.</Text>
            : <Text>Keine positive Varianzzunahme vorhanden; ein Anteil kann nicht berechnet werden.</Text>}
          <Text size="sm" c="dimmed">Mean: {result.valid_pairs} von {result.rows.length} Paaren mit positiver Varianzzunahme.</Text>
          <Image src={varianceRoiArtifactUrl(runId, saved.id, result.plot, projectId)} alt="ROI-Auswertung: Gesamtbilder mit Auswahl und zugehörige Ausschnitte" fit="contain" />
          <Button component="a" w="fit-content" href={varianceRoiArtifactUrl(runId, saved.id, result.plot, projectId, true)}>ROI-Plot als PNG herunterladen</Button>
          <Table><Table.Thead><Table.Tr><Table.Th>Unruhe</Table.Th><Table.Th>ROI-Flächenanteil (%)</Table.Th><Table.Th>Anteil der Varianzzunahme (%)</Table.Th></Table.Tr></Table.Thead>
            <Table.Tbody>{result.rows.map(row => <Table.Tr key={row.label}><Table.Td>{row.label}</Table.Td><Table.Td>{fmt(row.area_percent)}</Table.Td><Table.Td>{fmt(row.increase_percent)}</Table.Td></Table.Tr>)}
              <Table.Tr fw={600}><Table.Td>Mean</Table.Td><Table.Td>{fmt(result.area_percent)}</Table.Td><Table.Td>{fmt(result.mean_increase_percent)}</Table.Td></Table.Tr>
            </Table.Tbody></Table>
          <Button component="a" w="fit-content" href={varianceRoiArtifactUrl(runId, saved.id, result.table, projectId, true)}>Tabelle als PNG herunterladen</Button>
        </>}
      </>}
    </>}
  </Stack></Paper>;
}
