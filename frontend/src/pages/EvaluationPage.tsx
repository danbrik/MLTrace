import { Alert, Badge, Button, Center, Checkbox, Collapse, Group, Loader, MultiSelect, NumberInput, Paper, ScrollArea, SegmentedControl, Select, SimpleGrid, Stack, Table, Text, TextInput, Title } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { ArrowLeft, Calculator, ChevronDown, ChevronRight, Plus, RefreshCw, Save, Search, SlidersHorizontal, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Shape } from 'plotly.js';
import { activateWorkspaceDrift, calculateWorkspaceDrift, calculateWorkspaceSeparation, deleteDriftLayout, deleteSeparationLayout, deleteWorkspaceDriftCalculation, deleteWorkspaceSeparationResult, getEvaluationScorePreview, getEvaluationWorkspace, listDriftLayouts, listEvaluationWorkspaceModels, listEvaluationWorkspaceRuns, listSeparationLayouts, listWorkspaceDriftCalculations, listWorkspaceSeparationResults, previewWorkspaceDrift, saveDriftLayout, saveSeparationLayout, setWorkspaceDriftBucketIncluded, setWorkspaceSeparationIncluded } from '../api';
import { DateTime24Input } from '../components/DateTime24Input';
import { PlotlyChart, type PlotlyChartSelection } from '../components/PlotlyChart';
import { StepCard } from '../components/StepCard';
import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from '../components/TablePagination';
import { facetOption } from '../testing/facetFilters';
import { countModelFacetValues, modelMatchesFilters, type ModelFilterMetadata, type ModelFilterState } from '../testing/modelFilters';
import { WorkspaceRunPicker } from '../evaluation/WorkspaceRunPicker';
import { normalizeHorizontalSelection, normalizePairs, separationPairsEqual } from '../evaluation/workspaceHelpers';
import type { EvaluationDriftCalculation, EvaluationDriftLayout, EvaluationDriftLayoutPayload, EvaluationDriftPreview, EvaluationScorePreview, EvaluationSeparationLayout, EvaluationSeparationPair, EvaluationWorkspaceModel, EvaluationWorkspaceSeparationResult, TestingRun } from '../types';

type View = 'overview' | 'separation' | 'drift';
const msg = (error: unknown) => error instanceof Error ? error.message : 'Unknown error';
const fail = (title: string, error: unknown) => notifications.show({ color: 'red', title, message: msg(error) });
const metric = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? 'N/A' : value.toLocaleString(undefined, { maximumFractionDigits: 5 });
const key = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
const range = normalizeHorizontalSelection;

function Summary({ model }: { model: EvaluationWorkspaceModel }) {
  return <Paper withBorder p="md"><Group justify="space-between" mb="sm"><div><Text fw={700}>{model.name}</Text><Text size="xs" c="dimmed">Artifact {model.artifact_signature.slice(0, 12)}…</Text></div><Badge>{model.method_type}</Badge></Group><Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Sep_median</Table.Th><Table.Th>Sep_min</Table.Th><Table.Th>D_mean</Table.Th><Table.Th>D_max</Table.Th><Table.Th>Included A rows</Table.Th><Table.Th>Active B run</Table.Th></Table.Tr></Table.Thead><Table.Tbody><Table.Tr><Table.Td>{metric(model.sep_median)}</Table.Td><Table.Td>{metric(model.sep_min)}</Table.Td><Table.Td>{metric(model.d_mean)}</Table.Td><Table.Td>{metric(model.d_max)}</Table.Td><Table.Td>{model.included_separation_results}</Table.Td><Table.Td>{model.active_drift_testing_run_id ? `#${model.active_drift_testing_run_id}` : 'N/A'}</Table.Td></Table.Tr></Table.Tbody></Table></Paper>;
}

function ScorePlot({ preview, shapes, loading, onSelect }: { preview: EvaluationScorePreview | null; shapes: Shape[]; loading: boolean; onSelect: (selection: PlotlyChartSelection) => void }) {
  if (loading) return <Paper withBorder p="xl"><Center><Group gap="sm"><Loader size="sm" /><Text size="sm">Loading score plot…</Text></Group></Center></Paper>;
  if (!preview) return <Alert color="gray">Select a finished inference run and press “Load score plot”.</Alert>;
  return <Stack gap="xs">{preview.decimated && <Badge variant="light" w="fit-content">Preview {preview.points.length.toLocaleString()} / {preview.total.toLocaleString()} points</Badge>}<PlotlyChart data={[{ type: 'scattergl', mode: 'lines', x: preview.points.map((p) => p.timestamp), y: preview.points.map((p) => p.value), line: { width: 1.2 } }]} layout={{ dragmode: 'select', hovermode: 'x unified', shapes, uirevision: `${preview.testing_run_id}:${preview.score_series}`, showlegend: false, xaxis: { type: 'date', rangeslider: { visible: true, thickness: .1 }, title: { text: 'Dataset-local time' } }, yaxis: { title: { text: 'Anomaly score' } } }} config={{ scrollZoom: true, modeBarButtonsToAdd: ['select2d'] }} height={430} onSelected={onSelect} /><Text size="xs" c="dimmed">Zoom, pan and the range slider only change the view. Drag horizontally to apply the selected range tool.</Text></Stack>;
}

/** Score preview loading shared by both methods: manual first load, automatic on a score series switch. */
function useScorePlot(runId: number | null, score: string) {
  const [preview, setPreview] = useState<EvaluationScorePreview | null>(null), [loading, setLoading] = useState(false);
  const attempted = useRef<string | null>(null);
  const load = useCallback(async () => {
    if (!runId) return;
    attempted.current = score;
    setLoading(true);
    try { setPreview(await getEvaluationScorePreview(runId, { score_series: score, max_points: 8000 })); }
    catch (e) { fail('Could not load the score plot', e); }
    finally { setLoading(false); }
  }, [runId, score]);
  // Once a plot is on screen a score switch refreshes it; a failed attempt is not retried in a loop.
  useEffect(() => { if (preview && !loading && attempted.current !== score) void load(); }, [load, loading, preview, score]);
  const clear = useCallback(() => { attempted.current = null; setPreview(null); }, []);
  return { preview, clear, loading, load };
}

function Separation({ model, runs, onSummary }: { model: EvaluationWorkspaceModel; runs: TestingRun[]; onSummary: (value: EvaluationWorkspaceModel) => void }) {
  const [runId, setRunId] = useState<number | null>(null), [score, setScore] = useState('score');
  const [layouts, setLayouts] = useState<EvaluationSeparationLayout[]>([]);
  const [layoutId, setLayoutId] = useState<number | null>(null), [layoutName, setLayoutName] = useState('');
  const [pairs, setPairs] = useState<EvaluationSeparationPair[]>([]), [pairKey, setPairKey] = useState<string | null>(null), [selected, setSelected] = useState<string[]>([]);
  const [tool, setTool] = useState<'normal' | 'anomaly'>('normal'), [results, setResults] = useState<EvaluationWorkspaceSeparationResult[]>([]), [busy, setBusy] = useState(false);
  const { preview, clear: clearPlot, loading: plotLoading, load: loadPlot } = useScorePlot(runId, score);
  const run = runs.find((item) => item.id === runId) ?? null, pair = pairs.find((item) => item.pair_key === pairKey) ?? null;
  const persistedLayout = layouts.find((item) => item.id === layoutId) ?? null;
  const layoutDirty = !persistedLayout || persistedLayout.name !== layoutName || !separationPairsEqual(persistedLayout.pairs, pairs);
  const reload = useCallback(async () => setResults(await listWorkspaceSeparationResults(model.training_run_id)), [model.training_run_id]);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => { if (!run) { setLayouts([]); return; } void listSeparationLayouts(run.training_dataset_id).then(setLayouts).catch((e) => fail('Could not load A layouts', e)); }, [run]);
  const update = (values: Partial<EvaluationSeparationPair>) => pairKey && setPairs((items) => items.map((item) => item.pair_key === pairKey ? { ...item, ...values } : item));
  const shapes = useMemo<Shape[]>(() => pairs.flatMap((item) => [{ type: 'rect', xref: 'x', yref: 'paper', x0: item.normal_start, x1: item.normal_end, y0: 0, y1: 1, fillcolor: 'rgba(34,139,230,.13)', line: { color: '#228be6' }, layer: 'below' } as Shape, { type: 'rect', xref: 'x', yref: 'paper', x0: item.anomaly_start, x1: item.anomaly_end, y0: 0, y1: 1, fillcolor: 'rgba(250,82,82,.14)', line: { color: '#fa5252' }, layer: 'below' } as Shape]).filter((item) => Boolean(item.x0 && item.x1)), [pairs]);
  function adopt(row: EvaluationSeparationLayout) { setLayoutId(row.id); setLayoutName(row.name); setPairs(row.pairs); setPairKey((current) => row.pairs.some((item) => item.pair_key === current) ? current : row.pairs[0]?.pair_key ?? null); setSelected((current) => { const keys = row.pairs.map((item) => item.pair_key); const kept = keys.filter((item) => current.includes(item)); return kept.length ? kept : keys; }); }
  function choose(value: string | null) { const row = layouts.find((item) => item.id === Number(value)); if (!row) { setLayoutId(null); setLayoutName(''); setPairs([]); setPairKey(null); setSelected([]); return; } setLayoutId(row.id); setLayoutName(row.name); setPairs(row.pairs); setPairKey(row.pairs[0]?.pair_key ?? null); setSelected(row.pairs.map((item) => item.pair_key)); }
  function add() { const id = key('pair'), item = { pair_key: id, name: `Pair ${pairs.length + 1}`, normal_start: '', normal_end: '', anomaly_start: '', anomaly_end: '' }; setPairs((rows) => [...rows, item]); setPairKey(id); setSelected((rows) => [...rows, id]); }
  function remove(target: string) { setPairs((rows) => { const next = rows.filter((row) => row.pair_key !== target); setPairKey((current) => current === target ? next[0]?.pair_key ?? null : current); return next; }); setSelected((rows) => rows.filter((id) => id !== target)); }
  async function save() { if (!run) return; setBusy(true); try { const row = await saveSeparationLayout({ training_dataset_id: run.training_dataset_id, name: layoutName, pairs: normalizePairs(pairs) }, layoutId ?? undefined); adopt(row); setLayouts(await listSeparationLayouts(run.training_dataset_id)); notifications.show({ color: 'green', message: `Layout v${row.version} saved.` }); } catch (e) { fail('Could not save layout', e); } finally { setBusy(false); } }
  async function calculate() { const keys = selected.filter((id) => pairs.some((item) => item.pair_key === id)); if (!runId || !layoutId || !keys.length) return; if (layoutDirty) { notifications.show({ color: 'orange', message: 'Save the visible A layout before calculating it.' }); return; } setBusy(true); try { onSummary(await calculateWorkspaceSeparation(model.training_run_id, { testing_run_id: runId, layout_id: layoutId, pair_keys: keys, score_series: score })); await reload(); } catch (e) { fail('A calculation failed', e); } finally { setBusy(false); } }
  return <Stack gap="md"><StepCard index={1} title="Inference run" color="blue"><WorkspaceRunPicker runs={runs} runId={runId} score={score} loading={plotLoading} loaded={Boolean(preview)} onLoadPlot={() => void loadPlot()} onRun={(id) => { setRunId(id); setScore('score'); clearPlot(); choose(null); }} onScore={setScore} /></StepCard><StepCard index={2} title="Dataset-wide A layout" color="violet" action={<Button variant="light" leftSection={<Plus size={15} />} disabled={!run} onClick={add}>Add pair</Button>}><Stack gap="sm"><SimpleGrid cols={{ base: 1, md: 2 }}><Select label="Saved layout" clearable searchable data={layouts.map((item) => ({ value: String(item.id), label: `${item.name} · v${item.version}` }))} value={layoutId ? String(layoutId) : null} onChange={choose} /><TextInput label="Layout name" value={layoutName} onChange={(e) => setLayoutName(e.currentTarget.value)} /></SimpleGrid>{pairs.length > 0 && <Select label="Pair to edit" data={pairs.map((item) => ({ value: item.pair_key, label: item.name }))} value={pairKey} onChange={setPairKey} />}{pair && <><TextInput label="Pair name" value={pair.name} onChange={(e) => update({ name: e.currentTarget.value })} /><SegmentedControl value={tool} onChange={(value) => setTool(value as 'normal' | 'anomaly')} data={[{ value: 'normal', label: 'Normalbereich wählen' }, { value: 'anomaly', label: 'Anomaliebereich wählen' }]} /><SimpleGrid cols={{ base: 1, md: 2, xl: 4 }}><DateTime24Input label="Normal start" value={pair.normal_start} onChange={(value) => update({ normal_start: value })} /><DateTime24Input label="Normal end (exclusive)" value={pair.normal_end} onChange={(value) => update({ normal_end: value })} /><DateTime24Input label="Anomaly start" value={pair.anomaly_start} onChange={(value) => update({ anomaly_start: value })} /><DateTime24Input label="Anomaly end (inclusive)" value={pair.anomaly_end} onChange={(value) => update({ anomaly_end: value })} /></SimpleGrid></>}<ScorePlot preview={preview} shapes={shapes} loading={plotLoading} onSelect={(value) => { const r = range(value); update(tool === 'normal' ? { normal_start: r.start, normal_end: r.end } : { anomaly_start: r.start, anomaly_end: r.end }); }} /><ScrollArea><Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Use</Table.Th><Table.Th>Name</Table.Th><Table.Th>Normal [start,end)</Table.Th><Table.Th>Anomaly [start,end]</Table.Th><Table.Th /></Table.Tr></Table.Thead><Table.Tbody>{pairs.map((item) => <Table.Tr key={item.pair_key}><Table.Td><Checkbox checked={selected.includes(item.pair_key)} onChange={(e) => setSelected((rows) => e.currentTarget.checked ? [...new Set([...rows, item.pair_key])] : rows.filter((id) => id !== item.pair_key))} /></Table.Td><Table.Td>{item.name}</Table.Td><Table.Td>{item.normal_start || '—'} → {item.normal_end || '—'}</Table.Td><Table.Td>{item.anomaly_start || '—'} → {item.anomaly_end || '—'}</Table.Td><Table.Td><Button size="compact-xs" color="red" variant="subtle" onClick={() => remove(item.pair_key)}><Trash2 size={14} /></Button></Table.Td></Table.Tr>)}{!pairs.length && <Table.Tr><Table.Td colSpan={5}><Text ta="center" c="dimmed">No pair in this layout. Add one, or save the empty layout.</Text></Table.Td></Table.Tr>}</Table.Tbody></Table></ScrollArea><Group justify="flex-end">{layoutId && <Button color="red" variant="subtle" onClick={() => void deleteSeparationLayout(layoutId).then(() => choose(null)).then(() => run && listSeparationLayouts(run.training_dataset_id).then(setLayouts))}>Delete layout</Button>}<Button variant="default" leftSection={<Save size={15} />} loading={busy} disabled={!run || !layoutName.trim()} onClick={() => void save()}>Save layout</Button><Button leftSection={<Calculator size={15} />} loading={busy} disabled={!layoutId || !selected.length} onClick={() => void calculate()}>Calculate selected pairs</Button></Group></Stack></StepCard><Paper withBorder p="md"><Title order={4} mb="sm">A results across inference runs</Title><ScrollArea><Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Include</Table.Th><Table.Th>Run</Table.Th><Table.Th>Pair</Table.Th><Table.Th>Median</Table.Th><Table.Th>MAD</Table.Th><Table.Th>Scale</Table.Th><Table.Th>Sep_i</Table.Th><Table.Th>Sep95_i</Table.Th><Table.Th>Status</Table.Th><Table.Th /></Table.Tr></Table.Thead><Table.Tbody>{results.map((item) => <Table.Tr key={item.id}><Table.Td><Checkbox checked={item.included} disabled={item.stale} onChange={(e) => void setWorkspaceSeparationIncluded(model.training_run_id, item.id, e.currentTarget.checked).then(onSummary).then(reload)} /></Table.Td><Table.Td>#{item.testing_run_id}</Table.Td><Table.Td>{item.pair_name}</Table.Td><Table.Td>{metric(item.normal_median)}</Table.Td><Table.Td>{metric(item.normal_mad)}</Table.Td><Table.Td>{metric(item.robust_scale)}</Table.Td><Table.Td>{metric(item.separation)}</Table.Td><Table.Td>{metric(item.separation_p95)}</Table.Td><Table.Td><Badge color={item.stale ? 'orange' : 'green'}>{item.stale ? 'Stale' : 'Current'}</Badge></Table.Td><Table.Td><Button size="compact-xs" color="red" variant="subtle" onClick={() => void deleteWorkspaceSeparationResult(model.training_run_id, item.id).then(onSummary).then(reload)}><Trash2 size={14} /></Button></Table.Td></Table.Tr>)}{!results.length && <Table.Tr><Table.Td colSpan={10}><Text ta="center" c="dimmed">No A results yet.</Text></Table.Td></Table.Tr>}</Table.Tbody></Table></ScrollArea></Paper></Stack>;
}

const blankDrift = (dataset: number): EvaluationDriftLayoutPayload => ({ training_dataset_id: dataset, name: '', description: null, reference_start: '', reference_end: '', analysis_start: '', analysis_end: '', bucket_seconds: 86400, reference_exclusion_action: 'filter_points', exclusions: [], buckets: [] });

const driftPayload = (row: EvaluationDriftLayout): EvaluationDriftLayoutPayload => ({ training_dataset_id: row.training_dataset_id, name: row.name, description: row.description ?? null, reference_start: row.reference_start, reference_end: row.reference_end, analysis_start: row.analysis_start, analysis_end: row.analysis_end, bucket_seconds: row.bucket_seconds, reference_exclusion_action: row.reference_exclusion_action, exclusions: row.exclusions, buckets: row.buckets });

function Drift({ model, runs, onSummary }: { model: EvaluationWorkspaceModel; runs: TestingRun[]; onSummary: (value: EvaluationWorkspaceModel) => void }) {
  const [runId, setRunId] = useState<number | null>(null), [score, setScore] = useState('score');
  const [layouts, setLayouts] = useState<EvaluationDriftLayout[]>([]), [layoutId, setLayoutId] = useState<number | null>(null), [draft, setDraft] = useState<EvaluationDriftLayoutPayload>(blankDrift(0));
  const [tool, setTool] = useState<'reference' | 'analysis' | 'exclusion'>('reference'), [buckets, setBuckets] = useState<EvaluationDriftPreview | null>(null), [history, setHistory] = useState<EvaluationDriftCalculation[]>([]), [removed, setRemoved] = useState(false), [busy, setBusy] = useState(false);
  const { preview, clear: clearPlot, loading: plotLoading, load: loadPlot } = useScorePlot(runId, score);
  const run = runs.find((item) => item.id === runId) ?? null;
  const persistedLayout = layouts.find((item) => item.id === layoutId) ?? null;
  const layoutDirty = !persistedLayout || JSON.stringify(driftPayload(persistedLayout)) !== JSON.stringify(draft);
  const reload = useCallback(async () => setHistory(await listWorkspaceDriftCalculations(model.training_run_id)), [model.training_run_id]);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => { if (!run) { setLayouts([]); return; } void listDriftLayouts(run.training_dataset_id).then(setLayouts).catch((e) => fail('Could not load B layouts', e)); }, [run]);
  function choose(value: string | null) { const row = layouts.find((item) => item.id === Number(value)); setLayoutId(row?.id ?? null); setDraft(row ? driftPayload(row) : blankDrift(run?.training_dataset_id ?? 0)); setBuckets(null); }
  const shapes = useMemo<Shape[]>(() => [{ type: 'rect', xref: 'x', yref: 'paper', x0: draft.reference_start, x1: draft.reference_end, y0: 0, y1: 1, fillcolor: 'rgba(64,192,87,.14)', line: { color: '#40c057' }, layer: 'below' } as Shape, { type: 'rect', xref: 'x', yref: 'paper', x0: draft.analysis_start, x1: draft.analysis_end, y0: 0, y1: 1, fillcolor: 'rgba(34,139,230,.08)', line: { color: '#228be6' }, layer: 'below' } as Shape, ...draft.exclusions.map((item) => ({ type: 'rect', xref: 'x', yref: 'paper', x0: item.start_timestamp, x1: item.end_timestamp, y0: 0, y1: 1, fillcolor: 'rgba(134,142,150,.28)', line: { color: '#868e96' }, layer: 'below' } as Shape))].filter((item) => Boolean(item.x0 && item.x1)), [draft]);
  async function inspect(next = draft) { if (!runId) return; setBusy(true); try { setBuckets(await previewWorkspaceDrift(model.training_run_id, runId, score, next)); } catch (e) { fail('Bucket validation failed', e); } finally { setBusy(false); } }
  function decide(id: string, decision: 'include' | 'drop_bucket' | 'filter_points') { const definitions = (buckets?.buckets ?? []).map((item) => ({ bucket_key: item.bucket_key, start_timestamp: item.start_timestamp, end_timestamp: item.end_timestamp, decision: item.bucket_key === id ? decision : item.decision })); const next = { ...draft, buckets: definitions }; setDraft(next); void inspect(next); }
  async function save() { setBusy(true); try { const row = await saveDriftLayout(draft, layoutId ?? undefined); setLayoutId(row.id); setDraft(driftPayload(row)); if (run) setLayouts(await listDriftLayouts(run.training_dataset_id)); notifications.show({ color: 'green', message: `Layout v${row.version} saved.` }); } catch (e) { fail('Could not save layout', e); } finally { setBusy(false); } }
  async function calculate() { if (!runId || !layoutId) return; if (layoutDirty) { notifications.show({ color: 'orange', message: 'Save the visible B layout before calculating it.' }); return; } setBusy(true); try { onSummary(await calculateWorkspaceDrift(model.training_run_id, runId, layoutId, score)); await reload(); } catch (e) { fail('B calculation failed', e); } finally { setBusy(false); } }
  const shown = buckets?.buckets.filter((item) => item.status !== 'removed') ?? [], hidden = buckets?.buckets.filter((item) => item.status === 'removed') ?? [], active = history.find((item) => item.active) ?? null;
  return <Stack gap="md"><StepCard index={1} title="Inference run" color="blue"><WorkspaceRunPicker runs={runs} runId={runId} score={score} loading={plotLoading} loaded={Boolean(preview)} onLoadPlot={() => void loadPlot()} onRun={(id) => { setRunId(id); setScore('score'); clearPlot(); const row = runs.find((item) => item.id === id); setLayoutId(null); setDraft(blankDrift(row?.training_dataset_id ?? 0)); setBuckets(null); }} onScore={setScore} /></StepCard><StepCard index={2} title="Dataset-wide B layout" color="violet" action={<Button variant="light" leftSection={<RefreshCw size={15} />} loading={busy} disabled={!run} onClick={() => void inspect()}>Generate / validate buckets</Button>}><Stack gap="sm"><SimpleGrid cols={{ base: 1, md: 2 }}><Select label="Saved layout" clearable searchable data={layouts.map((item) => ({ value: String(item.id), label: `${item.name} · v${item.version}` }))} value={layoutId ? String(layoutId) : null} onChange={choose} /><TextInput label="Layout name" value={draft.name} onChange={(e) => setDraft((value) => ({ ...value, name: e.currentTarget.value }))} /></SimpleGrid><SegmentedControl value={tool} onChange={(value) => setTool(value as typeof tool)} data={[{ value: 'reference', label: 'Reference range' }, { value: 'analysis', label: 'Analysis range' }, { value: 'exclusion', label: 'Add exclusion' }]} /><SimpleGrid cols={{ base: 1, md: 2, xl: 4 }}><DateTime24Input label="Reference start" value={draft.reference_start} onChange={(value) => setDraft((row) => ({ ...row, reference_start: value }))} /><DateTime24Input label="Reference end" value={draft.reference_end} onChange={(value) => setDraft((row) => ({ ...row, reference_end: value }))} /><DateTime24Input label="Analysis start" value={draft.analysis_start} onChange={(value) => setDraft((row) => ({ ...row, analysis_start: value }))} /><DateTime24Input label="Analysis end" value={draft.analysis_end} onChange={(value) => setDraft((row) => ({ ...row, analysis_end: value }))} /></SimpleGrid><SimpleGrid cols={{ base: 1, md: 2 }}><NumberInput label="Bucket duration L (hours)" min={.001} value={draft.bucket_seconds / 3600} onChange={(value) => setDraft((row) => ({ ...row, bucket_seconds: Math.max(Number(value) || .001, .001) * 3600 }))} /><Select label="Reference/exclusion conflict" value={draft.reference_exclusion_action} onChange={(value) => setDraft((row) => ({ ...row, reference_exclusion_action: (value ?? 'filter_points') as 'filter_points' | 'drop_reference' }))} data={[{ value: 'filter_points', label: 'Filter excluded reference points' }, { value: 'drop_reference', label: 'Remove reference (blocks calculation)' }]} /></SimpleGrid><ScorePlot preview={preview} shapes={shapes} loading={plotLoading} onSelect={(selection) => { const r = range(selection); if (tool === 'reference') setDraft((row) => ({ ...row, reference_start: r.start, reference_end: r.end })); else if (tool === 'analysis') setDraft((row) => ({ ...row, analysis_start: r.start, analysis_end: r.end })); else setDraft((row) => ({ ...row, exclusions: [...row.exclusions, { exclusion_key: key('exclusion'), name: `Exclusion ${row.exclusions.length + 1}`, start_timestamp: r.start, end_timestamp: r.end }] })); }} />{draft.exclusions.map((item, index) => <SimpleGrid key={item.exclusion_key} cols={{ base: 1, md: 4 }}><TextInput label="Exclusion" value={item.name} onChange={(e) => setDraft((row) => ({ ...row, exclusions: row.exclusions.map((x, i) => i === index ? { ...x, name: e.currentTarget.value } : x) }))} /><DateTime24Input label="Start" value={item.start_timestamp} onChange={(value) => setDraft((row) => ({ ...row, exclusions: row.exclusions.map((x, i) => i === index ? { ...x, start_timestamp: value } : x) }))} /><DateTime24Input label="End (inclusive)" value={item.end_timestamp} onChange={(value) => setDraft((row) => ({ ...row, exclusions: row.exclusions.map((x, i) => i === index ? { ...x, end_timestamp: value } : x) }))} /><Button mt={25} color="red" variant="subtle" onClick={() => setDraft((row) => ({ ...row, exclusions: row.exclusions.filter((_, i) => i !== index) }))}>Remove</Button></SimpleGrid>)}{buckets && <><Paper withBorder p="sm"><Group justify="space-between"><Text fw={600}>Reference (not part of D_mean/D_max)</Text><Badge color={buckets.near_zero_iqr ? 'orange' : 'green'}>IQR {metric(buckets.reference_iqr)}</Badge></Group><Text size="sm">{draft.reference_start} → {draft.reference_end} · {buckets.reference_point_count}/{buckets.reference_original_point_count} points</Text></Paper><ScrollArea><Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Bucket</Table.Th><Table.Th>Start</Table.Th><Table.Th>End</Table.Th><Table.Th>Used/original</Table.Th><Table.Th>Warning</Table.Th><Table.Th>Decision</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{shown.map((item) => <Table.Tr key={item.bucket_key}><Table.Td>{item.bucket_key}</Table.Td><Table.Td>{item.start_timestamp}</Table.Td><Table.Td>{item.end_timestamp}</Table.Td><Table.Td>{item.used_point_count}/{item.original_point_count}</Table.Td><Table.Td><Badge color={item.status === 'ready' ? 'green' : item.status === 'conflict' ? 'orange' : 'gray'}>{item.reason ?? item.status}</Badge></Table.Td><Table.Td><Select size="xs" value={item.decision} disabled={['continuity gap', 'incomplete remainder', 'overlaps reference'].includes(item.reason ?? '')} data={[{ value: 'include', label: 'Include' }, { value: 'drop_bucket', label: 'Remove bucket' }, { value: 'filter_points', label: 'Filter excluded points' }]} onChange={(value) => decide(item.bucket_key, (value ?? 'include') as 'include' | 'drop_bucket' | 'filter_points')} /></Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea>{hidden.length > 0 && <><Button variant="subtle" color="gray" rightSection={removed ? <ChevronDown size={14} /> : <ChevronRight size={14} />} onClick={() => setRemoved((value) => !value)}>Removed buckets ({hidden.length})</Button><Collapse in={removed}>{hidden.map((item) => <Group key={item.bucket_key} justify="space-between" p="xs"><Text size="sm">{item.start_timestamp} → {item.end_timestamp}</Text><Button size="compact-xs" onClick={() => decide(item.bucket_key, 'include')}>Restore</Button></Group>)}</Collapse></>}</>}<Group justify="flex-end">{layoutId && <Button color="red" variant="subtle" onClick={() => void deleteDriftLayout(layoutId).then(() => choose(null)).then(() => run && listDriftLayouts(run.training_dataset_id).then(setLayouts))}>Delete layout</Button>}<Button variant="default" leftSection={<Save size={15} />} loading={busy} disabled={!run || !draft.name.trim()} onClick={() => void save()}>Save layout</Button><Button leftSection={<Calculator size={15} />} loading={busy} disabled={!layoutId || !buckets || buckets.buckets.some((item) => item.status === 'conflict')} onClick={() => void calculate()}>Calculate Score Stability</Button></Group></Stack></StepCard><Paper withBorder p="md"><Title order={4} mb="sm">B calculation history</Title>{active && <PlotlyChart data={[{ type: 'scatter', mode: 'lines+markers', x: active.buckets.filter((item) => item.included && item.normalized_drift != null).map((item) => item.start_timestamp), y: active.buckets.filter((item) => item.included && item.normalized_drift != null).map((item) => item.normalized_drift), name: 'D_k' }]} layout={{ xaxis: { type: 'date', title: { text: 'Bucket start' } }, yaxis: { title: { text: 'Normalized Wasserstein distance D_k' } } }} height={320} />}<ScrollArea><Table striped withTableBorder><Table.Thead><Table.Tr><Table.Th>Active</Table.Th><Table.Th>Run</Table.Th><Table.Th>Created</Table.Th><Table.Th>D_mean</Table.Th><Table.Th>D_max</Table.Th><Table.Th>IQR</Table.Th><Table.Th>Status</Table.Th><Table.Th /></Table.Tr></Table.Thead><Table.Tbody>{history.map((item) => <Table.Tr key={item.id}><Table.Td><Checkbox checked={item.active} disabled={item.stale} onChange={() => void activateWorkspaceDrift(model.training_run_id, item.id).then(onSummary).then(reload)} /></Table.Td><Table.Td>#{item.testing_run_id}</Table.Td><Table.Td>{item.created_at.replace('T', ' ')}</Table.Td><Table.Td>{metric(item.d_mean)}</Table.Td><Table.Td>{metric(item.d_max)}</Table.Td><Table.Td>{metric(item.reference_iqr)}</Table.Td><Table.Td><Badge color={item.stale ? 'orange' : item.active ? 'green' : 'gray'}>{item.stale ? 'Stale' : item.active ? 'Active' : 'History'}</Badge></Table.Td><Table.Td><Button size="compact-xs" color="red" variant="subtle" onClick={() => void deleteWorkspaceDriftCalculation(model.training_run_id, item.id).then(onSummary).then(reload)}><Trash2 size={14} /></Button></Table.Td></Table.Tr>)}{!history.length && <Table.Tr><Table.Td colSpan={8}><Text ta="center" c="dimmed">No B calculations yet.</Text></Table.Td></Table.Tr>}</Table.Tbody></Table></ScrollArea>{active && <><Title order={5} mt="md">Active calculation buckets</Title><ScrollArea><Table withTableBorder><Table.Thead><Table.Tr><Table.Th>Include</Table.Th><Table.Th>Bucket</Table.Th><Table.Th>Used/original</Table.Th><Table.Th>W₁</Table.Th><Table.Th>D_k</Table.Th><Table.Th>Reason</Table.Th></Table.Tr></Table.Thead><Table.Tbody>{active.buckets.map((item) => <Table.Tr key={item.id}><Table.Td><Checkbox checked={item.included} disabled={item.status !== 'ready' || active.stale} onChange={(e) => void setWorkspaceDriftBucketIncluded(model.training_run_id, item.id, e.currentTarget.checked).then(onSummary).then(reload)} /></Table.Td><Table.Td>{item.start_timestamp} → {item.end_timestamp}</Table.Td><Table.Td>{item.used_point_count}/{item.original_point_count}</Table.Td><Table.Td>{metric(item.wasserstein_1)}</Table.Td><Table.Td>{metric(item.normalized_drift)}</Table.Td><Table.Td>{item.reason ?? '—'}</Table.Td></Table.Tr>)}</Table.Tbody></Table></ScrollArea></>}</Paper></Stack>;
}

function modelMetadata(model: EvaluationWorkspaceModel): ModelFilterMetadata {
  return {
    datasetIds: model.training_dataset_names,
    preprocessingId: model.preprocessing_pipeline_name || null,
    methodId: model.method_type || null,
    inputResolution: null,
    searchableValues: [model.name, model.method_type, model.method_family, model.preprocessing_pipeline_name, ...model.training_dataset_names],
  };
}

function ModelPicker({ models, loading, selectedId, onSelect }: { models: EvaluationWorkspaceModel[]; loading: boolean; selectedId: number | null; onSelect: (id: number) => void }) {
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [datasets, setDatasets] = useState<string[]>([]), [preprocessings, setPreprocessings] = useState<string[]>([]), [methods, setMethods] = useState<string[]>([]);
  const [page, setPage] = useState(1);

  const metadata = useMemo(() => models.map(modelMetadata), [models]);
  const filterState = useMemo<ModelFilterState>(
    () => ({ query: search, datasetIds: datasets, preprocessingIds: preprocessings, methodIds: methods, requiredInputResolution: null }),
    [datasets, methods, preprocessings, search],
  );
  const filtered = useMemo(() => models.filter((_, index) => modelMatchesFilters(metadata[index], filterState)), [filterState, metadata, models]);
  const counts = useMemo(() => ({
    dataset: countModelFacetValues(metadata, filterState, 'dataset'),
    preprocessing: countModelFacetValues(metadata, filterState, 'preprocessing'),
    method: countModelFacetValues(metadata, filterState, 'method'),
  }), [filterState, metadata]);
  const datasetOptions = useMemo(() => [...new Set(models.flatMap((item) => item.training_dataset_names))]
    .map((value) => facetOption(value, value, counts.dataset, datasets))
    .sort((left, right) => left.label.localeCompare(right.label)), [counts.dataset, datasets, models]);
  const preprocessingOptions = useMemo(() => [...new Set(models.map((item) => item.preprocessing_pipeline_name).filter(Boolean))]
    .map((value) => facetOption(value, value, counts.preprocessing, preprocessings))
    .sort((left, right) => left.label.localeCompare(right.label)), [counts.preprocessing, models, preprocessings]);
  const methodOptions = useMemo(() => [...new Set(models.map((item) => item.method_type).filter(Boolean))]
    .map((value) => facetOption(value, value.replaceAll('_', ' '), counts.method, methods))
    .sort((left, right) => left.label.localeCompare(right.label)), [counts.method, methods, models]);
  const paged = useMemo(
    () => filtered.slice((page - 1) * DEFAULT_TABLE_PAGE_SIZE, page * DEFAULT_TABLE_PAGE_SIZE),
    [filtered, page],
  );
  const activeFilters = Boolean(search.trim()) || datasets.length > 0 || preprocessings.length > 0 || methods.length > 0;
  const reset = () => { setSearch(''); setDatasets([]); setPreprocessings([]); setMethods([]); };

  useEffect(() => setPage(1), [search, datasets, preprocessings, methods]);
  useEffect(() => {
    setPage((current) => Math.min(current, Math.max(1, Math.ceil(filtered.length / DEFAULT_TABLE_PAGE_SIZE))));
  }, [filtered.length]);

  return (
    <StepCard index={1} title="Select trained model artifact" subtitle="Exactly one model can be evaluated at a time." color="blue">
      <Stack gap="sm">
        <Group justify="space-between" wrap="wrap">
          <Group gap="xs">
            <Button
              variant="subtle"
              size="compact-sm"
              leftSection={<SlidersHorizontal size={16} />}
              rightSection={filtersOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              onClick={() => setFiltersOpen((open) => !open)}
            >
              Filters
            </Button>
            <Badge variant="light">{filtered.length} matching model{filtered.length === 1 ? '' : 's'}</Badge>
            {loading && <Loader size="xs" />}
          </Group>
          {activeFilters && <Button variant="subtle" color="gray" size="compact-sm" onClick={reset}>Reset filters</Button>}
        </Group>
        <Collapse in={filtersOpen}>
          <Stack gap="sm">
            <TextInput
              placeholder="Search models, methods and training datasets"
              leftSection={<Search size={16} />}
              value={search}
              onChange={(event) => setSearch(event.currentTarget.value)}
            />
            <SimpleGrid cols={{ base: 1, sm: 3 }}>
              <MultiSelect label="Training datasets" data={datasetOptions} value={datasets} onChange={setDatasets} searchable clearable />
              <MultiSelect label="Preprocessing" data={preprocessingOptions} value={preprocessings} onChange={setPreprocessings} searchable clearable />
              <MultiSelect label="Methods" data={methodOptions} value={methods} onChange={setMethods} searchable clearable />
            </SimpleGrid>
            <Text size="xs" c="dimmed">Counts show the models that would remain after the search and all other filter categories. Values inside one category use OR.</Text>
          </Stack>
        </Collapse>
        <Paper withBorder radius="sm">
          <ScrollArea>
            <Table striped highlightOnHover verticalSpacing="sm" miw={900}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Model</Table.Th>
                  <Table.Th>Method</Table.Th>
                  <Table.Th>Training datasets</Table.Th>
                  <Table.Th>Preprocessing</Table.Th>
                  <Table.Th>Artifact</Table.Th>
                  <Table.Th />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {paged.map((item) => {
                  const selected = selectedId === item.training_run_id;
                  return (
                    <Table.Tr key={`${item.training_run_id}:${item.artifact_signature}`}>
                      <Table.Td>{item.name}</Table.Td>
                      <Table.Td>{item.method_type}</Table.Td>
                      <Table.Td>{item.training_dataset_names.join(', ') || '—'}</Table.Td>
                      <Table.Td>{item.preprocessing_pipeline_name}</Table.Td>
                      <Table.Td><Text ff="monospace" size="xs">{item.artifact_signature.slice(0, 12)}…</Text></Table.Td>
                      <Table.Td>
                        <Group justify="flex-end">
                          <Button
                            size="compact-sm"
                            variant={selected ? 'filled' : 'light'}
                            color={selected ? 'green' : 'blue'}
                            onClick={() => onSelect(item.training_run_id)}
                          >
                            {selected ? 'Selected' : 'Use'}
                          </Button>
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  );
                })}
                {!loading && filtered.length === 0 && (
                  <Table.Tr>
                    <Table.Td colSpan={6}>
                      <Stack align="center" gap="xs" py="md">
                        <Text size="sm" c="dimmed">No finished signed model artifact matches the combined filters.</Text>
                        {activeFilters && <Button variant="light" size="compact-sm" onClick={reset}>Reset filters</Button>}
                      </Stack>
                    </Table.Td>
                  </Table.Tr>
                )}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        </Paper>
        <TablePagination totalItems={filtered.length} page={page} onChange={setPage} />
      </Stack>
    </StepCard>
  );
}

export function EvaluationPage({ active }: { active: boolean }) {
  const [models, setModels] = useState<EvaluationWorkspaceModel[]>([]), [selectedId, setSelectedId] = useState<number | null>(null), [runs, setRuns] = useState<TestingRun[]>([]), [view, setView] = useState<View>('overview'), [loading, setLoading] = useState(false);
  const selected = models.find((item) => item.training_run_id === selectedId) ?? null;
  const refresh = useCallback(async () => { setLoading(true); try { const rows = await listEvaluationWorkspaceModels(); setModels(rows); if (selectedId) { const summary = await getEvaluationWorkspace(selectedId); setModels((items) => items.map((item) => item.training_run_id === selectedId ? { ...item, ...summary } : item)); } } catch (e) { fail('Could not load evaluation models', e); } finally { setLoading(false); } }, [selectedId]);
  useEffect(() => { if (active) void refresh(); }, [active, refresh]);
  useEffect(() => { if (selectedId) void listEvaluationWorkspaceRuns(selectedId).then(setRuns).catch((e) => fail('Could not load inference runs', e)); else setRuns([]); }, [selectedId]);
  const merge = (summary: EvaluationWorkspaceModel) => setModels((items) => items.map((item) => item.training_run_id === summary.training_run_id ? { ...item, ...summary } : item));
  if (!active) return null;
  return <Stack gap="lg"><Group justify="space-between"><div><Title order={2}>Evaluation</Title><Text c="dimmed">Model-centred Event Separation and Score Stability.</Text></div><Button variant="default" leftSection={<RefreshCw size={15} />} loading={loading} onClick={() => void refresh()}>Refresh</Button></Group>{view !== 'overview' && selected ? <><Group><Button variant="subtle" leftSection={<ArrowLeft size={16} />} onClick={() => { setView('overview'); void refresh(); }}>Back to model overview</Button><Badge>{selected.name}</Badge></Group><Summary model={selected} />{view === 'separation' ? <Separation model={selected} runs={runs} onSummary={merge} /> : <Drift model={selected} runs={runs} onSummary={merge} />}</> : <><ModelPicker models={models} loading={loading} selectedId={selectedId} onSelect={setSelectedId} />{selected && <><Summary model={selected} /><SimpleGrid cols={{ base: 1, md: 2 }}><Paper withBorder p="lg"><Title order={4}>A · Event Separation</Title><Text c="dimmed" size="sm" my="sm">Pair normal and anomaly ranges, reuse dataset layouts and aggregate included results across runs.</Text><Button fullWidth onClick={() => setView('separation')}>Open Event Separation</Button></Paper><Paper withBorder p="lg"><Title order={4}>B · Score Stability</Title><Text c="dimmed" size="sm" my="sm">Define reference, fixed buckets and exclusions, then manage calculation history.</Text><Button fullWidth onClick={() => setView('drift')}>Open Score Stability</Button></Paper></SimpleGrid></>}</>}</Stack>;
}
