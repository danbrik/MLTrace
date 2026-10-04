import { Alert, Button, Group, Image, Loader, NumberInput, Paper, Progress, Select, Slider, Stack, Table, Text, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import { abortVarianceRoiJob, evaluateVarianceRoi, getVarianceRoiState, prepareVarianceRoi, varianceRoiArtifactUrl, varianceRoiPreviewUrl } from '../api';
import { RoiEditor } from './RoiEditor';
import { orientedRoi, roiCoordinates, roiValidation } from './roiGeometry';
import { displayTime } from './helpers';
import type { RoiDraft, RoiState, RoiResult } from './roiTypes';
const fmt = (value: number | null) => value === null ? '—' : value.toLocaleString('de-DE', {maximumFractionDigits: 2});
const restored = (result: RoiResult): RoiDraft => ({roi: orientedRoi(result.roi), opacity: result.opacity,
  heatmap_mode: result.heatmap_mode ?? 'global', sensitivity: result.sensitivity ?? 1});

export function VarianceRoiPanel({runId, projectId, active}: {runId: number; projectId: string; active: boolean}) {
  const [state, setState] = useState<RoiState | null>(null);
  const [config, setConfig] = useState<RoiDraft | null>(null);
  const [angleInput, setAngleInput] = useState<string | number>(0);
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
      setConfig(result ? restored(result) : {roi: {version: 2, center_x: basis.width / 2, center_y: basis.height / 2, width: basis.width, height: basis.height, angle_degrees: 0}, opacity: .5, heatmap_mode: 'local', sensitivity: 1});
      setEditing(!result);
    } else if (!editing && result) {
      setConfig(restored(result));
    }
  }, [!!basis, saved?.id, editing]);
  useEffect(() => {if (config) setAngleInput(config.roi.angle_degrees);}, [config?.roi.angle_degrees, editing]);
  const angleValid = angleInput !== '' && Number.isFinite(Number(angleInput));
  const validation = !angleValid ? 'Bitte einen gültigen Winkel eingeben.' : basis && config ? roiValidation(config.roi, basis.width, basis.height) : null;
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
        <Text size="sm">Heatmap-Deckkraft: {Math.round(config.opacity * 100)} %</Text>
        <Slider thumbLabel="Heatmap-Deckkraft" value={config.opacity * 100} min={0} max={100} disabled={busy || running} onChange={value => setConfig({...config, opacity: value / 100})} />
        <Text size="sm">Empfindlichkeit: {fmt(config.sensitivity)}</Text>
        <Slider thumbLabel="Empfindlichkeit" value={config.sensitivity} min={.25} max={4} step={.05} precision={2} disabled={busy || running} onChange={sensitivity => setConfig({...config, sensitivity})} />
        <Text size="sm" c="dimmed">Rot zeigt Varianzzunahme, Blau Varianzabnahme. Ohne Änderung bleibt das Hintergrundbild unverändert. Höhere Empfindlichkeit hebt schwache Änderungen hervor; die Tabelle bleibt unverändert.</Text>
        <Group align="end">
          <NumberInput label="Winkel (°)" description="Positive Winkel drehen im Uhrzeigersinn." value={angleInput} step={1} disabled={busy || running}
            onChange={value => {setAngleInput(value); if (value !== '' && Number.isFinite(Number(value))) setConfig({...config, roi: {...config.roi, angle_degrees: Number(value)}});}} />
          <Button variant="light" disabled={busy || running} onClick={() => {setAngleInput(0); setConfig({...config, roi: {...config.roi, angle_degrees: 0}});}}>Drehung zurücksetzen</Button>
        </Group>
        {config.roi.width === basis.width && config.roi.height === basis.height && <Text size="sm" c="dimmed">Bei einer Vollbildauswahl bitte vor dem Drehen das Rechteck verkleinern, damit es innerhalb des Bildes bleibt.</Text>}
        <Text size="sm" c="dimmed">Der Ausschnitt wird mit Nearest Neighbor gerade ausgerichtet, ohne Glättung oder Mischung benachbarter Pixelwerte. Die Tabelle verwendet die Originalpixel.</Text>
        {validation && <Alert color="red">{validation}</Alert>}
        <RoiEditor invalid={!!validation} key={`${projectId}-${runId}-${pair}`} width={basis.width} height={basis.height} value={config.roi} disabled={busy || running}
          previewUrl={varianceRoiPreviewUrl(runId, Number(pair), projectId, config)}
          onChange={roi => setConfig({...config, roi})} />
        <Text size="sm">{roiCoordinates(config.roi)}</Text>
        <Group><Button disabled={busy || running || !!validation} onClick={() => void action(() => evaluateVarianceRoi(runId, config, projectId), true)}>Fertig</Button>
          {result && <Button variant="subtle" disabled={busy || running} onClick={() => setEditing(false)}>Bearbeitung verwerfen</Button>}
        </Group>
      </> : <>
        <Button w="fit-content" variant="light" disabled={busy || running} onClick={() => {setConfig({...config, heatmap_mode: 'local'}); setEditing(true);}}>ROI bearbeiten</Button>
        {result && saved && <>
          {running && <Text size="sm" c="dimmed">Das bisher gespeicherte Ergebnis bleibt bis zum erfolgreichen Abschluss sichtbar.</Text>}
          <Text size="sm">{roiCoordinates(result.roi)} · Heatmap-Deckkraft: {Math.round(result.opacity * 100)} % · {result.heatmap_mode === 'local' ? `Empfindlichkeit: ${fmt(result.sensitivity ?? 1)}` : 'Bisherige flächige Darstellung'}</Text>
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
