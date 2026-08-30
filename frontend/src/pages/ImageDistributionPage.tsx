import { Alert, Badge, Button, Group, Loader, Paper, Select, SimpleGrid, Stack, Text, Title } from '@mantine/core';
import { notifications } from '@mantine/notifications';
import { Download, Play } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import {
  calculateImageDistribution,
  imageDistributionCsvUrl,
  listDatasets,
  listPreprocessingPipelines,
} from '../api';
import { PlotlyChart } from '../components/PlotlyChart';
import { withLineGapPolicy } from '../lib/plotGaps';
import type { Data, Layout } from '../lib/plotly';
import type {
  Dataset,
  ImageDistributionHourlyPoint,
  ImageDistributionMetricSummary,
  ImageDistributionResult,
  PreprocessingPipeline,
} from '../types';

type Props = { active: boolean };
type MetricKey = 'mean_intensity' | 'spatial_std_intensity' | 'q95_intensity';

const PERIOD_COLORS: Record<string, string> = {
  train: 'rgba(34, 139, 230, 0.10)',
  validation: 'rgba(64, 192, 87, 0.12)',
  test: 'rgba(250, 176, 5, 0.12)',
  mixed: 'rgba(132, 94, 247, 0.10)',
};
const PERIOD_BADGE_COLORS: Record<string, string> = {
  train: 'blue',
  validation: 'green',
  test: 'yellow',
  mixed: 'violet',
};

function parseDatasetTime(value: string): number {
  return Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(value) ? value : `${value}Z`);
}

function continuity(points: ImageDistributionHourlyPoint[]): number[] {
  let segment = 0;
  return points.map((point, index) => {
    if (index > 0 && parseDatasetTime(point.hour) - parseDatasetTime(points[index - 1].hour) !== 3_600_000) segment += 1;
    return segment;
  });
}

function periodShapes(result: ImageDistributionResult): NonNullable<Partial<Layout>['shapes']> {
  return result.periods.map((period) => ({
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: period.start,
    x1: period.end,
    y0: 0,
    y1: 1,
    fillcolor: PERIOD_COLORS[period.usage_label] ?? 'rgba(134, 142, 150, 0.10)',
    line: { width: 0 },
    layer: 'below',
  }));
}

function metricTraces(points: ImageDistributionHourlyPoint[], key: MetricKey): Data[] {
  const x = points.map((point) => point.hour);
  const segments = continuity(points);
  const metric = (point: ImageDistributionHourlyPoint): ImageDistributionMetricSummary => point[key];
  const hover = points.map((point) => `${point.image_count} Bild${point.image_count === 1 ? '' : 'er'}`);
  return [
    withLineGapPolicy({
      type: 'scatter', mode: 'lines', x, y: points.map((point) => metric(point).q25),
      line: { width: 0 }, hoverinfo: 'skip', showlegend: false,
    } as Data, { continuity: segments }),
    withLineGapPolicy({
      type: 'scatter', mode: 'lines', x, y: points.map((point) => metric(point).q75),
      line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(34, 139, 230, 0.20)',
      name: 'Q25–Q75', hoverinfo: 'skip',
    } as Data, { continuity: segments }),
    withLineGapPolicy({
      type: 'scatter', mode: 'lines', x, y: points.map((point) => metric(point).median),
      line: { color: '#228be6', width: 2 }, name: 'Stündlicher Median',
      text: hover, hovertemplate: '%{x}<br>Median: %{y:.5g}<br>%{text}<extra></extra>',
    } as Data, { continuity: segments }),
  ];
}

function MetricPlot({ result, metricKey, title, yTitle }: {
  result: ImageDistributionResult;
  metricKey: MetricKey;
  title: string;
  yTitle: string;
}) {
  return (
    <Paper withBorder p="md">
      <Title order={4} mb="xs">{title}</Title>
      <PlotlyChart
        data={metricTraces(result.hourly, metricKey)}
        height={360}
        rescaleYOnVisibleX
        layout={{
          xaxis: { title: { text: 'Zeit' }, type: 'date', rangeslider: { visible: true } },
          yaxis: { title: { text: yTitle } },
          hovermode: 'x unified',
          legend: { orientation: 'h' },
          shapes: periodShapes(result),
          margin: { l: 72, r: 24, t: 12, b: 62 },
        }}
      />
    </Paper>
  );
}

export function ImageDistributionPage({ active }: Props) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [pipelines, setPipelines] = useState<PreprocessingPipeline[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [result, setResult] = useState<ImageDistributionResult | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [calculating, setCalculating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    setLoadingOptions(true);
    Promise.all([listDatasets(), listPreprocessingPipelines()])
      .then(([nextDatasets, nextPipelines]) => {
        if (cancelled) return;
        setDatasets(nextDatasets);
        setPipelines(nextPipelines);
      })
      .catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason)); })
      .finally(() => { if (!cancelled) setLoadingOptions(false); });
    return () => { cancelled = true; };
  }, [active]);

  const datasetOptions = useMemo(() => datasets.map((dataset) => ({
    value: String(dataset.id),
    label: dataset.status === 'ready' ? dataset.name : `${dataset.name} (${dataset.status})`,
    disabled: dataset.status !== 'ready',
  })), [datasets]);
  const pipelineOptions = useMemo(() => pipelines.map((pipeline) => ({ value: String(pipeline.id), label: pipeline.name })), [pipelines]);

  async function run() {
    if (!datasetId || !pipelineId) return;
    setCalculating(true);
    setError(null);
    setResult(null);
    try {
      const next = await calculateImageDistribution(Number(datasetId), Number(pipelineId));
      setResult(next);
      notifications.show({
        color: next.failed_images ? 'yellow' : 'green',
        title: next.cache_hit ? 'Gespeicherte Analyse geladen' : 'Analyse abgeschlossen',
        message: `${next.successful_images.toLocaleString('de-DE')} von ${next.total_images.toLocaleString('de-DE')} Bildern ausgewertet.`,
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setCalculating(false);
    }
  }

  return (
    <Stack gap="md">
      <div>
        <Title order={2}>Zeitliche Bildverteilung</Title>
        <Text c="dimmed">Beobachte Änderungen der gesamten Intensitätsverteilung und ihres oberen Bereichs über den vollständigen Datensatz.</Text>
      </div>

      <Paper withBorder p="md">
        <Stack gap="md">
          <SimpleGrid cols={{ base: 1, md: 2 }}>
            <Select label="Datensatz" placeholder="Datensatz auswählen" data={datasetOptions} value={datasetId} onChange={setDatasetId} searchable disabled={loadingOptions || calculating} />
            <Select label="Preprocessing" placeholder="Pipeline auswählen" data={pipelineOptions} value={pipelineId} onChange={setPipelineId} searchable disabled={loadingOptions || calculating} />
          </SimpleGrid>
          <Group>
            <Button leftSection={calculating ? <Loader size={16} color="white" /> : <Play size={16} />} onClick={run} disabled={!datasetId || !pipelineId || calculating}>
              {calculating ? 'Alle Bilder werden verarbeitet …' : 'Analyse starten'}
            </Button>
            <Text size="sm" c="dimmed">Gleiche, unveränderte Konfigurationen werden direkt aus der CSV geladen.</Text>
          </Group>
        </Stack>
      </Paper>

      {error && <Alert color="red" title="Analyse nicht möglich">{error}</Alert>}

      {result && (
        <>
          <Paper withBorder p="md">
            <Group justify="space-between" align="flex-start">
              <div>
                <Group gap="xs">
                  <Badge color={result.cache_hit ? 'blue' : 'green'}>{result.cache_hit ? 'Aus CSV geladen' : 'Neu berechnet'}</Badge>
                  <Text fw={600}>{result.dataset_name} · {result.preprocessing_pipeline_name}</Text>
                </Group>
                <Text size="sm" c="dimmed" mt={6}>
                  {result.successful_images.toLocaleString('de-DE')} Bilder, {result.hourly.length.toLocaleString('de-DE')} Stundenfenster. Pixelwerte werden in der von der Pipeline ausgegebenen Skala ausgewertet.
                </Text>
                {result.failed_images > 0 && <Text size="sm" c="orange" mt={4}>{result.failed_images.toLocaleString('de-DE')} Bilder konnten nicht verarbeitet werden; Details stehen in der CSV.</Text>}
              </div>
              <Button component="a" href={imageDistributionCsvUrl(result.cache_key)} leftSection={<Download size={16} />} variant="light">CSV herunterladen</Button>
            </Group>
            {result.periods.length > 0 && (
              <Group gap="xs" mt="md">
                <Text size="sm" fw={600}>Markierte Zeiträume:</Text>
                {Array.from(new Set(result.periods.map((period) => period.usage_label))).map((label) => (
                  <Badge key={label} variant="light" color={PERIOD_BADGE_COLORS[label] ?? 'gray'}>
                    {label} ({result.periods.filter((period) => period.usage_label === label).length})
                  </Badge>
                ))}
                <Text size="xs" c="dimmed">Die transparenten Flächen zeigen die gespeicherten Train-/Validierungs-/Testbereiche.</Text>
              </Group>
            )}
          </Paper>

          {result.hourly.length === 0 ? (
            <Alert color="yellow">Keine erfolgreich verarbeiteten Bilder für die Zeitaggregation vorhanden.</Alert>
          ) : (
            <Stack gap="md">
              <MetricPlot result={result} metricKey="mean_intensity" title="Mittlere Bildintensität über die Zeit" yTitle="Mittlere Pixelintensität" />
              <MetricPlot result={result} metricKey="spatial_std_intensity" title="Räumliche Pixel-Standardabweichung über die Zeit" yTitle="Räumliche Standardabweichung" />
              <MetricPlot result={result} metricKey="q95_intensity" title="95%-Quantil der Pixelintensität über die Zeit" yTitle="Q95 der Pixelintensität" />
            </Stack>
          )}
        </>
      )}
    </Stack>
  );
}
