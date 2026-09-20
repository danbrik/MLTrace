import { useEffect, useMemo, useState } from 'react';
import { Accordion, Alert, Badge, Button, Code, Group, Loader, Paper, SegmentedControl, Select, Stack, Table, Text, Title } from '@mantine/core';
import type { Data } from '../lib/plotly';
import { sensorApi } from '../api';
import { PlotlyChart } from '../components/PlotlyChart';
import type { ResultRow, ResultSeries, SensorRun } from '../timeSeries/types';
import { loadResults } from '../timeSeries/resultLoading';

export function TimeSeriesResultsPage({ active }: { active: boolean }) {
  const [runs, setRuns] = useState<SensorRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [run, setRun] = useState<SensorRun>();
  const [series, setSeries] = useState<ResultSeries>();
  const [sensor, setSensor] = useState('0');
  const [subset, setSubset] = useState('test');
  const [space, setSpace] = useState('original');
  const [score, setScore] = useState('window_score');
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [runLoading, setRunLoading] = useState(false);
  const [runError, setRunError] = useState('');
  const [seriesError, setSeriesError] = useState('');
  const [runRetry, setRunRetry] = useState(0);
  const [seriesRetry, setSeriesRetry] = useState(0);
  useEffect(() => { if (active) sensorApi.runs().then(rows => setRuns(rows.filter(r => r.status === 'finished'))).catch(e => setError(String(e))); }, [active]);
  useEffect(() => {
    setRun(undefined); setSeries(undefined); setRunError(''); setSeriesError(''); setSensor('0'); setOffset(0);
    if (!active || !selected) { setRunLoading(false); return; }
    setRunLoading(true);
    return loadResults(signal => sensorApi.run(Number(selected), signal), {
      success: setRun, error: setRunError, settled: () => setRunLoading(false),
    });
  }, [active, selected, runRetry]);
  useEffect(() => {
    setSeries(undefined); setSeriesError('');
    if (!active || !run || run.id !== Number(selected)) { setLoading(false); return; }
    setLoading(true);
    return loadResults(signal => sensorApi.series(run.id, subset, Number(sensor), space === 'scaled', offset, signal), {
      success: setSeries, error: setSeriesError, settled: () => setLoading(false),
    });
  }, [active, selected, run, subset, sensor, space, offset, seriesRetry]);
  const traces = useMemo(() => {
    const rows: (ResultRow | null)[] = [];
    for (const row of series?.rows ?? []) { const previous = rows.at(-1); if (previous && previous.segment_id !== row.segment_id) rows.push(null); rows.push(row); }
    const x = rows.map(r => r?.timestamp ?? null);
    const trace = (key: 'original' | 'reconstruction' | 'cascade' | 'second', name: string, color: string): Data => ({ x, y: rows.map(r => r?.[key] ?? null), name, type: 'scatter', mode: 'lines', line: { color, width: 1.5 }, connectgaps: false, text: rows.map(r => r?.tags.join(', ') ?? ''), hovertemplate: '%{x}<br>%{y}<br>%{text}<extra>%{fullData.name}</extra>' });
    const raw: ({ timestamp: string; original: number; segment_id: number; warmup: boolean } | null)[] = [];
    for (const r of series?.raw ?? []) { const prev = raw.at(-1); if (prev && prev.segment_id !== r.segment_id) raw.push(null); raw.push(r); }
    const data: Data[] = [{ x: raw.map(r => r?.timestamp ?? null), y: raw.map(r => r?.original ?? null), type: 'scatter', mode: 'lines', name: 'Original', line: { color: '#748ffc', width: 1.5 }, connectgaps: false }, trace('reconstruction', 'Endpoint-Rekonstruktion · AE1 / μ', '#20c997')];
    data.push({ x: raw.filter(r => r?.warmup).map(r => r!.timestamp), y: raw.filter(r => r?.warmup).map(r => r!.original), type: 'scatter', mode: 'markers', name: 'Anlaufpunkte · ohne Ergebnis', marker: { color: '#868e96', size: 5 } });
    if (run?.kind === 'usad') { data.push(trace('cascade', 'Endpoint-Rekonstruktion · AE2(AE1)', '#ff922b')); data.push(trace('second', 'Endpoint-Rekonstruktion · AE2', '#be4bdb')); }
    if (run?.kind === 'lstm_vae') {
      data.push({ x, y: rows.map(r => r ? r.reconstruction - (r.std ?? 0) : null), type: 'scatter', mode: 'lines', line: { width: 0 }, showlegend: false, connectgaps: false });
      data.push({ x, y: rows.map(r => r ? r.reconstruction + (r.std ?? 0) : null), type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(32,201,151,.15)', name: 'μ ± 1 σ', connectgaps: false });
    }
    data.push({ x, y: rows.map(r => r ? (score === 'window_score' ? r.window_score : r.endpoint_score) : null), type: 'scatter', mode: 'lines', name: score === 'window_score' ? 'Window anomaly score' : 'Endpoint anomaly score', line: { color: '#fa5252' }, xaxis: 'x2', yaxis: 'y2', connectgaps: false, text: rows.map(r => r?.tags.join(', ') ?? ''), hovertemplate: '%{x}<br>%{y}<br>%{text}<extra>%{fullData.name}</extra>' });
    return data;
  }, [series, run?.kind, score]);
  if (!active) return null;
  const columns = run?.snapshot.preview.columns ?? [];
  return <Stack p="md"><Group justify="space-between"><Title order={2}>Results</Title><Button variant="subtle" onClick={() => sensorApi.runs().then(rows => setRuns(rows.filter(r => r.status === 'finished'))).catch(e => setError(String(e)))}>Läufe aktualisieren</Button></Group><Text c="dimmed">Rekonstruktionen, Scores und Repräsentationen aus demselben Checkpoint.</Text>{error && <Alert color="red">{error}</Alert>}
    <Select label="Abgeschlossener Lauf" searchable data={runs.map(r => ({ value: String(r.id), label: `#${r.id} · ${r.name} · ${r.kind}` }))} value={selected} onChange={setSelected} />
    {runLoading && <Group role="status"><Loader size="sm" /><Text>Laufkonfiguration und Trainingskurven werden geladen …</Text></Group>}
    {runError && <Alert color="red" title="Lauf konnte nicht geladen werden">{runError}<Button display="block" mt="xs" onClick={() => setRunRetry(v => v + 1)}>Erneut laden</Button></Alert>}
    {!runs.length && <Alert>Noch keine abgeschlossenen Läufe. Training kann unter Training Pipelines gestartet werden.</Alert>}
    {run && <><Paper withBorder p="md"><Stack gap="sm"><Group><Badge>{run.kind}</Badge><Text>Checkpoint: {run.checkpoint_selection}</Text><Text>Epoche {run.selected_epoch}</Text><Text>Metrik: {run.selected_metric?.toPrecision(6) ?? 'letzte Epoche'}</Text></Group><Text size="sm">Fenster: {run.snapshot.window_length} Samples @ {run.snapshot.preview.sampling_interval_seconds / 60} min · nominale Historie {run.snapshot.preview.nominal_history_seconds / 60} min · Zeitstempelspanne {run.snapshot.preview.timestamp_span_seconds / 60} min.</Text><Text size="sm">Die Rekonstruktion bei t stammt aus der letzten Position des vollständigen Eingabefensters. Der Window anomaly score berücksichtigt das gesamte Fenster; der Endpoint anomaly score nur dessen letzte Position. Scores bleiben im skalierten Raum.</Text><Text size="sm" c="dimmed">Anlaufpunkte haben keine Ergebnisse. Linien werden an Sequenzgrenzen getrennt; Tags stammen vom Fensterendpunkt. Ohne Schwellenwertentscheidung oder Glättung.</Text></Stack></Paper>
      <Group grow><Select label="Gruppe" value={subset} data={['test', 'validation', 'train']} onChange={v => { setSubset(v ?? 'test'); setOffset(0); }} /><Select label="Sensor" value={sensor} data={columns.map((c, i) => ({ value: String(i), label: c }))} onChange={v => { setSensor(v ?? '0'); setOffset(0); }} /><SegmentedControl value={space} onChange={setSpace} data={[{ value: 'original', label: 'Originaleinheiten' }, { value: 'scaled', label: 'Skaliert' }]} /></Group>
      <SegmentedControl value={score} onChange={setScore} data={[{ value: 'window_score', label: 'Window anomaly score' }, { value: 'endpoint_score', label: 'Endpoint anomaly score' }]} />
      {loading && <Group role="status"><Loader size="sm" /><Text>Gespeicherte Ergebnisse werden geladen · bis zu 5.000 Endpunkte. Keine erneute Modellberechnung.</Text></Group>}
      {seriesError && <Alert color="red" title="Ergebnisse konnten nicht geladen werden">{seriesError}<Button display="block" mt="xs" onClick={() => setSeriesRetry(v => v + 1)}>Ergebnisse erneut laden</Button></Alert>}
      {series && !series.rows.length && <Alert>Für diese Auswahl sind keine vollständigen Ergebnisfenster vorhanden.</Alert>}{series && <><PlotlyChart height={630} data={traces} layout={{ grid: { rows: 2, columns: 1, pattern: 'independent' }, xaxis: { type: 'date', matches: 'x2', showticklabels: false }, xaxis2: { type: 'date', title: { text: 'UTC-Zeit · gemeinsamer Zoom' } }, yaxis: { title: { text: columns[Number(sensor)] }, domain: [.4, 1] }, yaxis2: { title: { text: score === 'window_score' ? 'Window anomaly score' : 'Endpoint anomaly score' }, domain: [0, .27] }, legend: { orientation: 'h' }, uirevision: `${run.id}-${subset}-${offset}` }} /><Group justify="space-between"><Text size="sm">{series.rows.length} von {series.total} Endpunkten · Seite {Math.floor(offset / 5000) + 1}</Text><Group><Button size="xs" variant="light" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 5000))}>Vorherige 5000</Button><Button size="xs" variant="light" disabled={offset + series.rows.length >= series.total} onClick={() => setOffset(offset + 5000)}>Nächste 5000</Button></Group></Group><Group>{Array.from(new Set(series.rows.flatMap(r => r.tags))).map(tag => <Badge key={tag} variant="light">{tag}</Badge>)}</Group></>}
      <Group><Button component="a" href={sensorApi.exportUrl(run.id, `export.csv?sensor=${sensor}&scaled=${space === 'scaled'}`)} variant="light">Scores / Sensor als CSV</Button><Button component="a" href={sensorApi.exportUrl(run.id, 'representations.zip')} variant="light">Latente Vektoren + Metadaten</Button></Group>
      <Title order={3}>Trainingsverlauf</Title><PlotlyChart height={300} data={[{ x: run.metrics?.map(m => m.epoch), y: run.metrics?.map(m => m.train_loss), type: 'scatter', mode: 'lines+markers', name: 'Train loss' }, { x: run.metrics?.map(m => m.epoch), y: run.metrics?.map(m => m.val_loss), type: 'scatter', mode: 'lines+markers', name: 'Validation / Checkpoint-Metrik' }]} layout={{ xaxis: { title: { text: 'Epoche' } } }} />
      <Accordion variant="separated"><Accordion.Item value="scaler"><Accordion.Control>Train-Skalierung pro Sensor</Accordion.Control><Accordion.Panel><Table><Table.Thead><Table.Tr><Table.Th>Sensor</Table.Th><Table.Th>Minimum</Table.Th><Table.Th>Maximum</Table.Th><Table.Th>Nenner</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{columns.map((c, i) => <Table.Tr key={c}><Table.Td>{c}</Table.Td><Table.Td>{run.snapshot.preview.scaler.minimum[i]}</Table.Td><Table.Td>{run.snapshot.preview.scaler.maximum[i]}</Table.Td><Table.Td>{run.snapshot.preview.scaler.denominator[i]}</Table.Td></Table.Tr>)}</Table.Tbody></Table><Text size="sm">Kein Clipping. Bei konstanten Trainingsspalten ist der Nenner 1.</Text></Accordion.Panel></Accordion.Item><Accordion.Item value="snapshot"><Accordion.Control>Eingefrorene Konfiguration, Herkunft und Checkpoint</Accordion.Control><Accordion.Panel><Code block style={{ maxHeight: 550, overflow: 'auto' }}>{JSON.stringify({ snapshot: run.snapshot, checkpoint: run.checkpoint, results: run.result }, null, 2)}</Code></Accordion.Panel></Accordion.Item></Accordion>
    </>}
  </Stack>;
}
