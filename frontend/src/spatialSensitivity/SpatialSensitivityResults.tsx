import { Accordion, Alert, Button, Group, Image, Paper, ScrollArea, SimpleGrid, Stack, Table, Tabs, Text, Title } from '@mantine/core';
import { Download } from 'lucide-react';
import { spatialSensitivityArtifactUrl } from '../api';
import type { SpatialSensitivityResult } from '../types';
import { SPATIAL_COLUMNS, SPATIAL_METRICS, SPATIAL_VERSION } from './metrics';

const fmt = (value: unknown) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('de-DE', {maximumFractionDigits:3}) : '—';
const counts = ['normal_candidate_count', 'normal_image_count', 'normal_attempted_count', 'event_candidate_count', 'event_image_count', 'event_attempted_count'];

export function SpatialSensitivityResults({runId, result}: {runId:number; result:SpatialSensitivityResult}) {
  const url = (name:string) => spatialSensitivityArtifactUrl(runId, name);
  const download = <Button component="a" href={url('spatial_sensitivity_artifacts.zip')} leftSection={<Download size={16}/>}>Alle Artefakte</Button>;
  if (result.analysis_version !== SPATIAL_VERSION) return <Paper withBorder p="md"><Stack>
    <Alert color="yellow" title="Historisches Ergebnis">Dieses Ergebnis gehört zur früheren Berechnungsmethodik. Es wird nicht als Vier-Karten-Ergebnis interpretiert. Lade die Konfiguration und berechne sie erneut. Die unveränderten Originalartefakte bleiben verfügbar.</Alert>{download}
  </Stack></Paper>;
  return <Stack gap="md">
    <Paper withBorder p="md"><Group justify="space-between"><div><Title order={4}>Vier räumliche Änderungskarten</Title>
      <Text size="sm" c="dimmed">Normal ≤ {fmt(result.normal_sample_size)} · Ereignis ≤ {fmt(result.event_sample_size)} · Seed {fmt(result.sampling_seed)} · MAD-Epsilon = {result.epsilon}</Text>
      <Text size="sm" c="dimmed">Q95 über Eventframes · lineare Quantilinterpolation · Quotienten-Nenneruntergrenze {result.ratio_floor ?? 1e-12}</Text>
    </div>{download}</Group></Paper>
    {result.events.some(row => Number(row.normal_sample_shortfall) > 0 || Number(row.event_sample_shortfall) > 0) && <Alert color="yellow" title="Kleinere Stichprobe">Mindestens ein Fenster enthielt weniger valide Bilder als angefordert. Verwendet werden alle gefundenen validen Bilder bis zur gewählten Stichprobengröße.</Alert>}
    <SimpleGrid cols={{base:1,sm:2,lg:4}}>{SPATIAL_METRICS.map(({name,title}) => <Paper key={name} withBorder p="md" radius="md">
      <Text fw={600}>{name}</Text><Text size="xs" c="dimmed">{title}</Text>
      <Text mt="sm">Median Q: {fmt(result.median[`Q_${name}`])}</Text>
      {name.startsWith('D_') && <Text>Median P in: {fmt(result.median[`P_in_${name}`])} %</Text>}
      <Text size="xs" c="dimmed">Gemeinsame Farbskala: 0 … {fmt(result.vmax?.[name])}</Text>
    </Paper>)}</SimpleGrid>
    <Text size="sm" c="dimmed">Q vergleicht die mittlere Änderung pro Pixel innen/außen: 1 = gleich, 3 = dreifach innen (bei positivem Außensignal). Bei Außenwert 0 wird 10⁻¹² als Nenner verwendet; 0/0 wird dadurch 0, nicht als Gleichheit interpretiert. P in ist der Anteil der absoluten Gesamtänderung und hängt auch von der ROI-Fläche ab. Alle Kennzahlen bleiben ungeclippt.</Text>
    <Image loading="lazy" fit="contain" src={url('publication_overview.png')} alt="Übersicht der vier Änderungskarten"/>
    <Tabs defaultValue="D_med" keepMounted={false}><Tabs.List>{SPATIAL_METRICS.map(({name}) => <Tabs.Tab key={name} value={name}>{name}</Tabs.Tab>)}</Tabs.List>
      {SPATIAL_METRICS.map(({name,title,aggregate}) => <Tabs.Panel key={name} value={name} pt="md"><Stack>
        <Title order={4}>{title}</Title>
        <SimpleGrid cols={{base:1,md:2}}><Image loading="lazy" fit="contain" src={url(`events_grid_${name}.png`)} alt={`${name}: alle Ereignisse mit gemeinsamer Colorbar`}/><Image loading="lazy" fit="contain" src={url(`${aggregate}.png`)} alt={aggregate}/></SimpleGrid>
        <Image loading="lazy" fit="contain" src={url(`${name}_inside_outside.png`)} alt={`${name}: Mittelwerte innerhalb und außerhalb`}/>
        <Accordion variant="separated"><Accordion.Item value="individual"><Accordion.Control>Einzelereignis-Heatmaps</Accordion.Control><Accordion.Panel><SimpleGrid cols={{base:1,md:2}}>{result.events.map((event,index) => <div key={index}><Text>{String(event.event_id)} · {name}</Text><Image loading="lazy" fit="contain" src={url(`event_${String(index+1).padStart(3,'0')}_${name}.png`)} alt={`${event.event_id}: ${name}, vollständiges Originalbild mit ROI`}/></div>)}</SimpleGrid></Accordion.Panel></Accordion.Item></Accordion>
      </Stack></Tabs.Panel>)}
    </Tabs>
    <Group justify="space-between"><Title order={4}>Ereigniswerte und Median</Title><Button component="a" variant="light" href={url('results.csv')}>CSV herunterladen</Button></Group>
    <ScrollArea><Table striped miw={2600}><Table.Thead><Table.Tr>{['Event','Datensatz',...counts,...SPATIAL_COLUMNS].map(key => <Table.Th key={key}>{key}</Table.Th>)}</Table.Tr></Table.Thead><Table.Tbody>
      {[...result.events,result.median].map((row,index) => <Table.Tr key={index}><Table.Td>{String(row.event_id)}</Table.Td><Table.Td>{String(row.training_dataset ?? '')}</Table.Td>{[...counts,...SPATIAL_COLUMNS].map(key => <Table.Td key={key}>{fmt(row[key])}</Table.Td>)}</Table.Tr>)}
    </Table.Tbody></Table></ScrollArea>
  </Stack>;
}
