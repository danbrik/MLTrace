import { Alert, Button, ColorInput, Group, Loader, NumberInput, Pagination, Paper, Select, SimpleGrid, Stack, Switch, Table, Text, TextInput, Title } from '@mantine/core';
import { useEffect, useRef, useState } from 'react';
import Plotly from '../lib/plotly';
import { getTemporalDifferencePairs, getTemporalDifferenceSummary, saveTemporalDifferencePlot, temporalDifferenceCsvUrl } from '../api';
import { displayTime, labels, numberText, plotData, plotLayout, roles, validatePlot } from './helpers';
import type { PairPage, PlotSettings, Run, Summary } from './types';

function Chart({ summary, settings, runId }: { summary: Summary[]; settings: PlotSettings; runId: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const operations = useRef<Promise<unknown>>(Promise.resolve());
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const invalid = validatePlot(settings);
  useEffect(() => {
    const element = ref.current;
    if (!element || invalid) return;
    let cancelled = false;
    const work = operations.current.catch(() => undefined).then(async () => {
      if (cancelled) return;
      await Plotly.react(element, plotData(summary, settings), plotLayout(settings), { responsive: true, displaylogo: false, displayModeBar: false });
      if (!cancelled) setError(null);
    });
    operations.current = work;
    void work.catch(reason => { if (!cancelled) setError(String(reason)); });
    return () => { cancelled = true; };
  }, [summary, settings, invalid]);
  useEffect(() => {
    const element = ref.current!;
    let cancelled = false;
    const observer = new ResizeObserver(() => {
      operations.current = operations.current.catch(() => undefined).then(async () => {
        if (!cancelled && element.isConnected && element.classList.contains('js-plotly-plot')) await Plotly.Plots.resize(element);
      });
      void operations.current.catch(reason => { if (!cancelled) setError(String(reason)); });
    });
    observer.observe(element);
    return () => {
      cancelled = true; observer.disconnect();
      void operations.current.catch(() => undefined).then(() => { if (!element.isConnected) Plotly.purge(element); });
    };
  }, []);
  async function download(format: 'png' | 'svg') {
    if (!ref.current || invalid) return;
    setExporting(true); setError(null);
    try {
      await operations.current;
      const element = ref.current;
      const options = { format, filename: `temporal-difference-${runId}`, width: 1400, height: 700, scale: format === 'png' ? 2 : 1 };
      operations.current = Plotly.downloadImage(element, options);
      await operations.current;
    } catch (reason) { setError(String(reason)); }
    finally { setExporting(false); }
  }
  return <Stack>
    {invalid && <Alert color="orange">{invalid}</Alert>}
    {error && <Alert color="red">{error}</Alert>}
    <div ref={ref} style={{ width: '100%', height: 500, background: '#fff' }} />
    <Group><Button variant="light" disabled={!!invalid || exporting} onClick={() => void download('png')}>Plot als PNG</Button>
      <Button variant="light" disabled={!!invalid || exporting} onClick={() => void download('svg')}>Plot als SVG</Button></Group>
  </Stack>;
}

export function TemporalResults({ run, projectId, onPlotSaved }: { run: Run; projectId: string; onPlotSaved: (settings: PlotSettings) => void }) {
  const [summary, setSummary] = useState<Summary[] | null>(null);
  const [settings, setSettings] = useState<PlotSettings>(() => structuredClone(run.plot_settings));
  const [saved, setSaved] = useState(JSON.stringify(run.plot_settings));
  const [pairs, setPairs] = useState<PairPage | null>(null);
  const [page, setPage] = useState(1);
  const [role, setRole] = useState<string | null>(null), [delta, setDelta] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null), [pairError, setPairError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let cancelled = false;
    getTemporalDifferenceSummary(run.id, projectId).then(value => { if (!cancelled) { setSummary(value); setError(null); } })
      .catch(reason => { if (!cancelled) setError(String(reason)); });
    return () => { cancelled = true; };
  }, [run.id, projectId, retry]);
  useEffect(() => {
    let cancelled = false;
    setPairs(null); setPairError(null);
    getTemporalDifferencePairs(run.id, (page - 1) * 50, role, delta, projectId).then(value => { if (!cancelled) setPairs(value); })
      .catch(reason => { if (!cancelled) setPairError(String(reason)); });
    return () => { cancelled = true; };
  }, [run.id, projectId, page, role, delta, retry]);
  function update(values: Partial<PlotSettings>) { setSettings(current => ({ ...current, ...values })); }
  async function save() {
    setSaving(true); setError(null);
    const snapshot = structuredClone(settings);
    try { await saveTemporalDifferencePlot(run.id, snapshot, projectId); setSaved(JSON.stringify(snapshot)); onPlotSaved(snapshot); }
    catch (reason) { setError(String(reason)); }
    finally { setSaving(false); }
  }
  return <Stack>
    {error && <Alert color="red">{error}<Button variant="subtle" onClick={() => setRetry(value => value + 1)}>Erneut laden</Button></Alert>}
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>Ergebnisse</Title>
      <Text size="sm">{run.result?.width} × {run.result?.height} Pixel · {run.result?.total_pairs} Bildpaare · Median und Interquartilsbereich (Q1 bis Q3)</Text>
      <Text size="sm" c="dimmed">Die Ergebnisse sind gespeichert. Titel, Achsen und Farben lassen sich ohne erneute Bildberechnung ändern.</Text>
      {!summary ? <Loader size="sm" /> : <>
        <Table.ScrollContainer minWidth={1000}><Table striped><Table.Thead>
          <Table.Tr><Table.Th rowSpan={2}>Δt (s)</Table.Th>{roles.map(key => <Table.Th key={key} colSpan={5}>{labels[key]}</Table.Th>)}</Table.Tr>
          <Table.Tr>{roles.flatMap(key => ['Paare', 'Median', 'Q1', 'Q3', 'IQR'].map(label => <Table.Th key={`${key}-${label}`}>{label}</Table.Th>))}</Table.Tr>
        </Table.Thead><Table.Tbody>{run.config.deltas_seconds.map(value => <Table.Tr key={value}><Table.Td>{value}</Table.Td>
          {roles.flatMap(key => {
            const row = summary.find(item => item.role === key && item.delta_seconds === value);
            return [<Table.Td key={`${key}-count`}>{row?.pair_count || 'Keine Paare'}</Table.Td>, ...(['median', 'q1', 'q3', 'iqr'] as const).map(stat => <Table.Td key={`${key}-${stat}`}>{numberText(row?.[stat] ?? null)}</Table.Td>)];
          })}</Table.Tr>)}</Table.Tbody></Table></Table.ScrollContainer>
        <Group><Button component="a" variant="light" href={temporalDifferenceCsvUrl(run.id, 'summary', projectId)}>Ergebnistabelle als CSV</Button>
          <Button component="a" variant="light" href={temporalDifferenceCsvUrl(run.id, 'pairs', projectId)}>Alle Paarwerte als CSV</Button></Group>
        <Chart summary={summary} settings={settings} runId={run.id} />
      </>}
    </Stack></Paper>
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>Plot-Darstellung</Title>
      <SimpleGrid cols={{ base: 1, md: 3 }}>
        <TextInput label="Plot-Titel" value={settings.title} maxLength={250} onChange={event => update({ title: event.currentTarget.value })} />
        <TextInput label="X-Achsentitel" value={settings.x_title} maxLength={250} onChange={event => update({ x_title: event.currentTarget.value })} />
        <TextInput label="Y-Achsentitel" value={settings.y_title} maxLength={250} onChange={event => update({ y_title: event.currentTarget.value })} />
      </SimpleGrid>
      <SimpleGrid cols={{ base: 1, md: 2 }}>{(['x_range', 'y_range'] as const).map((key, index) => <Stack key={key}>
        <Switch label={`${index ? 'Y' : 'X'}-Achse: manuelle Grenzen`} checked={settings[key] !== null}
          onChange={event => update({ [key]: event.currentTarget.checked ? { minimum: 0, maximum: index ? Math.max(1, ...summary?.map(row => row.q3 ?? 0) ?? [1]) : Math.max(...run.config.deltas_seconds) } : null })} />
        {settings[key] && <Group grow>{(['minimum', 'maximum'] as const).map(bound => <NumberInput key={bound} label={bound === 'minimum' ? 'Untergrenze' : 'Obergrenze'}
          value={Number.isFinite(settings[key]![bound]) ? settings[key]![bound] : ''}
          onChange={value => update({ [key]: { ...settings[key]!, [bound]: value === '' ? NaN : Number(value) } })} />)}</Group>}
      </Stack>)}</SimpleGrid>
      <SimpleGrid cols={{ base: 1, md: 2 }}>{roles.map(key => <ColorInput key={key} label={`Farbe ${labels[key]}`} format="hex" value={settings[`${key}_color`]} onChange={value => update({ [`${key}_color`]: value })} />)}</SimpleGrid>
      <Group><Button disabled={saving || !!validatePlot(settings) || JSON.stringify(settings) === saved} onClick={() => void save()}>Darstellung speichern</Button>
        <Text size="sm" c="dimmed">{JSON.stringify(settings) === saved ? 'Darstellung gespeichert.' : 'Darstellung noch nicht gespeichert.'}</Text></Group>
    </Stack></Paper>
    <Paper withBorder p="lg"><Stack>
      <Title order={4}>Einzelne Paarwerte</Title>
      <Group><Select label="Zeitraum" placeholder="Beide Zeiträume" clearable value={role} data={roles.map(key => ({ value: key, label: labels[key] }))} onChange={value => { setRole(value); setPage(1); }} />
        <Select label="Zeitabstand" placeholder="Alle Abstände" clearable value={delta} data={run.config.deltas_seconds.map(value => ({ value: String(value), label: `${value} s` }))} onChange={value => { setDelta(value); setPage(1); }} /></Group>
      <Text size="sm" c="dimmed">Aufnahmezeiten in Europe/Berlin; UTC-Zeitpunkte und Bilddateien sind zusätzlich in der CSV enthalten.</Text>
      {pairError && <Alert color="red">{pairError}<Button variant="subtle" onClick={() => setRetry(value => value + 1)}>Erneut laden</Button></Alert>}
      {!pairs ? !pairError && <Loader size="sm" /> : <>
        <Table.ScrollContainer minWidth={650}><Table striped><Table.Thead><Table.Tr>{['Zeitraum', 'Δt (s)', 'Erstes Bild', 'Zweites Bild', 'Absolute Pixeländerung'].map(label => <Table.Th key={label}>{label}</Table.Th>)}</Table.Tr></Table.Thead>
          <Table.Tbody>{pairs.items.map(row => <Table.Tr key={row.id}><Table.Td>{labels[row.role]}</Table.Td><Table.Td>{row.delta_seconds}</Table.Td><Table.Td title={row.first_file}>{displayTime(row.first_timestamp)}</Table.Td><Table.Td title={row.second_file}>{displayTime(row.second_timestamp)}</Table.Td><Table.Td>{numberText(row.value)}</Table.Td></Table.Tr>)}</Table.Tbody></Table></Table.ScrollContainer>
        <Group><Pagination total={Math.max(1, Math.ceil(pairs.total / 50))} value={page} onChange={setPage} /><Text size="sm">{pairs.total} Paarwerte</Text></Group>
      </>}
    </Stack></Paper>
  </Stack>;
}
