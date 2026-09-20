import { useEffect, useState } from 'react';
import { Alert, Anchor, Badge, Button, Group, Modal, Paper, Select, Stack, Text, TextInput, Title } from '@mantine/core';
import { sensorApi } from '../api';
import { fieldDefaults, Flow, ModelFields } from '../timeSeries/ModelFields';
import type { ModelDefinition, Parameters, SensorModel } from '../timeSeries/types';

export function TimeSeriesModelsPage({ active }: { active: boolean }) {
  const [definitions, setDefinitions] = useState<ModelDefinition[]>([]);
  const [models, setModels] = useState<SensorModel[]>([]);
  const [id, setId] = useState<number>();
  const [kind, setKind] = useState('usad');
  const [name, setName] = useState('');
  const [config, setConfig] = useState<Parameters>({});
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const definition = definitions.find(d => d.kind === kind);
  function edit(model: SensorModel, clone = false) { setId(clone ? undefined : model.id); setKind(model.kind); setName(model.name + (clone ? ' – Kopie' : '')); setConfig(model.config); }
  useEffect(() => { if (active) Promise.all([sensorApi.definitions(), sensorApi.models()]).then(([defs, rows]) => { setDefinitions(defs); setModels(rows); }).catch(e => setError(String(e))); }, [active]);
  async function save() { setBusy(true); setError(''); try { const row = await sensorApi.saveModel({ name, kind, config }, id); setModels(await sensorApi.models()); edit(row); } catch (e) { setError(String(e)); } finally { setBusy(false); } }
  if (!active) return null;
  return <Stack p="md"><Group justify="space-between"><div><Title order={2}>Models</Title><Text c="dimmed">Nachvollziehbare Architekturvorlagen für einen gemeinsamen Fenstereingang.</Text></div><Button variant="light" disabled={!definitions.length} onClick={() => { setId(undefined); setName('Neue USAD-Variante'); setKind('usad'); setConfig(fieldDefaults(definitions.find(d => d.kind === 'usad')!.architecture)); }}>Neue Variante</Button></Group>
    {error && <Alert color="red" title="Aktion nicht möglich" withCloseButton onClose={() => setError('')}>{error}</Alert>}
    <Group align="stretch">{models.map(model => <Paper key={model.id} withBorder p="md" radius="md" style={{ flex: '1 1 250px' }}><Stack gap="xs"><Badge variant="light">{model.kind.toUpperCase()}</Badge><Text fw={600}>{model.name}</Text><Text size="sm" c="dimmed">{model.kind === 'tcn_ae' ? `Pooling ${model.config.pooling_factor} · MLTrace-Anpassung` : model.kind === 'lstm_vae' ? 'Adaptierte Fenster-Variante · Standardnormalprior' : 'Gemeinsamer Encoder · zwei Decoder'}</Text><Group><Button size="xs" onClick={() => edit(model)}>Anzeigen / bearbeiten</Button><Button size="xs" variant="subtle" onClick={() => edit(model, true)}>Duplizieren</Button></Group></Stack></Paper>)}</Group>
    {name && definition && <Paper withBorder p="lg" radius="md"><Stack><Group grow><TextInput label="Modellname" value={name} onChange={e => setName(e.currentTarget.value)} /><Select label="Architektur" disabled={!!id} data={definitions.map(d => ({ value: d.kind, label: d.label }))} value={kind} onChange={v => { if (v) { setKind(v); setConfig(fieldDefaults(definitions.find(d => d.kind === v)!.architecture)); } }} /></Group>
      <Flow steps={definition.diagram} /><Alert title="Architektur und Anpassungen">{definition.adaptations.map(note => <Text key={note} size="sm">{note}</Text>)}</Alert>
      <Anchor href={definition.source.url} target="_blank" size="sm">{definition.source.reference}</Anchor><ModelFields fields={definition.architecture} values={config} onChange={setConfig} />
      <Text size="sm" c="dimmed">L wird ausschließlich in der Pipeline festgelegt. Gemeinsamer MLTrace-Standard: nominal 3 Stunden, bei 5 Minuten Sampling L=36.</Text>
      <Group><Button loading={busy} disabled={!name.trim()} onClick={save}>Architektur speichern</Button>{id && <Button color="red" variant="subtle" onClick={() => setDeleting(true)}>Löschen</Button>}</Group>
    </Stack></Paper>}
    <Modal opened={deleting} onClose={() => setDeleting(false)} title="Modell löschen"><Stack><Text>„{name}“ löschen? Referenzierte Modelle bleiben geschützt.</Text><Button color="red" onClick={async () => { try { await sensorApi.deleteModel(id!); setModels(await sensorApi.models()); setName(''); setId(undefined); } catch (e) { setError(String(e)); } setDeleting(false); }}>Löschen</Button></Stack></Modal>
  </Stack>;
}
