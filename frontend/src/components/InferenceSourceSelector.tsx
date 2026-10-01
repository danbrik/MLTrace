import { Alert, Badge, Button, Checkbox, Collapse, Group, MultiSelect, Paper, ScrollArea, SimpleGrid, Stack, Table, Text, TextInput } from '@mantine/core';
import { ChevronDown, ChevronRight, Search, SlidersHorizontal } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { TestingRun } from '../types';
import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from './TablePagination';
import { countFacetValues, facetOption, matchingFacetRecords, type FacetFilterState } from '../testing/facetFilters';
import { inferenceFacetRecord } from '../testing/inferenceSource';
import { aggregationKeyForRun, aggregationLabel, metricKeyForRun, metricLabel, roiKeyForRun, roiLabelForRun } from '../testing/inferenceRunMetadata';

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return '—';
  const value = new Date(timestamp);
  return Number.isNaN(value.getTime()) ? timestamp : value.toLocaleString('de-DE', { hour12: false });
}

type Props = {
  runs: TestingRun[];
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  multiple?: boolean;
  disabled?: boolean;
};

export function InferenceSourceSelector({ runs, selectedIds, onSelectionChange, multiple = false, disabled = false }: Props) {
  const testingRuns = useMemo(() => runs.filter((run) => run.status === 'finished'), [runs]);
  const selectedRun = testingRuns.find((run) => String(run.id) === selectedIds[0]);
  const [sourceFiltersOpen, setSourceFiltersOpen] = useState(true);
  const [sourceSearch, setSourceSearch] = useState('');
  const [sourceModelFilters, setSourceModelFilters] = useState<string[]>([]);
  const [sourceTrainingDatasetFilters, setSourceTrainingDatasetFilters] = useState<string[]>([]);
  const [sourceDatasetFilters, setSourceDatasetFilters] = useState<string[]>([]);
  const [sourceRoiFilters, setSourceRoiFilters] = useState<string[]>([]);
  const [sourceMetricFilters, setSourceMetricFilters] = useState<string[]>([]);
  const [sourceAggregationFilters, setSourceAggregationFilters] = useState<string[]>([]);
  const [sourcePreprocessingFilters, setSourcePreprocessingFilters] = useState<string[]>([]);
  const [sourceMethodFilters, setSourceMethodFilters] = useState<string[]>([]);
  const [sourcePage, setSourcePage] = useState(1);
  const inferenceFacetRecords = useMemo(() => testingRuns.map(inferenceFacetRecord), [testingRuns]);
  const sourceFacetState = useMemo<FacetFilterState>(() => ({
    query: sourceSearch,
    selections: {
      model: sourceModelFilters,
      trainingDataset: sourceTrainingDatasetFilters,
      dataset: sourceDatasetFilters,
      roi: sourceRoiFilters,
      metric: sourceMetricFilters,
      aggregation: sourceAggregationFilters,
      preprocessing: sourcePreprocessingFilters,
      method: sourceMethodFilters,
    },
  }), [sourceAggregationFilters, sourceDatasetFilters, sourceMethodFilters, sourceMetricFilters, sourceModelFilters, sourcePreprocessingFilters, sourceRoiFilters, sourceSearch, sourceTrainingDatasetFilters]);
  const filteredRunRecords = useMemo(
    () => matchingFacetRecords(inferenceFacetRecords, sourceFacetState),
    [inferenceFacetRecords, sourceFacetState],
  );
  const testingRunById = useMemo(() => new Map(testingRuns.map((run) => [run.id, run])), [testingRuns]);
  const filteredRuns = useMemo(
    () => filteredRunRecords.map((record) => testingRunById.get(Number(record.id))).filter((run): run is TestingRun => Boolean(run)),
    [filteredRunRecords, testingRunById],
  );
  const pagedFilteredRuns = useMemo(
    () => filteredRuns.slice((sourcePage - 1) * DEFAULT_TABLE_PAGE_SIZE, sourcePage * DEFAULT_TABLE_PAGE_SIZE),
    [filteredRuns, sourcePage],
  );
  const sourceFacetCounts = useMemo(() => ({
    model: countFacetValues(inferenceFacetRecords, sourceFacetState, 'model'),
    trainingDataset: countFacetValues(inferenceFacetRecords, sourceFacetState, 'trainingDataset'),
    dataset: countFacetValues(inferenceFacetRecords, sourceFacetState, 'dataset'),
    roi: countFacetValues(inferenceFacetRecords, sourceFacetState, 'roi'),
    metric: countFacetValues(inferenceFacetRecords, sourceFacetState, 'metric'),
    aggregation: countFacetValues(inferenceFacetRecords, sourceFacetState, 'aggregation'),
    preprocessing: countFacetValues(inferenceFacetRecords, sourceFacetState, 'preprocessing'),
    method: countFacetValues(inferenceFacetRecords, sourceFacetState, 'method'),
  }), [inferenceFacetRecords, sourceFacetState]);
  const sourceModelOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [String(run.training_run_id), run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, sourceFacetCounts.model, sourceModelFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [sourceFacetCounts.model, sourceModelFilters, testingRuns]);
  const sourceTrainingDatasetOptions = useMemo(() => [...new Set(testingRuns.flatMap((run) => run.model_training_dataset_names ?? []))]
    .map((value) => facetOption(value, value, sourceFacetCounts.trainingDataset, sourceTrainingDatasetFilters))
    .sort((a, b) => a.label.localeCompare(b.label)), [sourceFacetCounts.trainingDataset, sourceTrainingDatasetFilters, testingRuns]);
  const sourceDatasetOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [String(run.training_dataset_id), run.training_dataset_name || `Inference dataset #${run.training_dataset_id}`]));
    return [...labels].map(([value, label]) => facetOption(value, label, sourceFacetCounts.dataset, sourceDatasetFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [sourceDatasetFilters, sourceFacetCounts.dataset, testingRuns]);
  const sourceRoiOptions = useMemo(() => {
    const labels = new Map(testingRuns.map((run) => [roiKeyForRun(run), roiLabelForRun(run)]));
    return [...labels].map(([value, label]) => facetOption(value, label, sourceFacetCounts.roi, sourceRoiFilters)).sort((a, b) => a.label.localeCompare(b.label));
  }, [sourceFacetCounts.roi, sourceRoiFilters, testingRuns]);
  const sourceMetricOptions = useMemo(() => [...new Set(testingRuns.map(metricKeyForRun))].map((value) => facetOption(value, metricLabel(value), sourceFacetCounts.metric, sourceMetricFilters)), [sourceFacetCounts.metric, sourceMetricFilters, testingRuns]);
  const sourceAggregationOptions = useMemo(() => [...new Set(testingRuns.map(aggregationKeyForRun))].map((value) => facetOption(value, aggregationLabel(value), sourceFacetCounts.aggregation, sourceAggregationFilters)), [sourceAggregationFilters, sourceFacetCounts.aggregation, testingRuns]);
  const sourcePreprocessingOptions = useMemo(() => [...new Set(testingRuns.map((run) => run.preprocessing_pipeline_name).filter(Boolean))].map((value) => facetOption(value, value, sourceFacetCounts.preprocessing, sourcePreprocessingFilters)).sort((a, b) => a.label.localeCompare(b.label)), [sourceFacetCounts.preprocessing, sourcePreprocessingFilters, testingRuns]);
  const sourceMethodOptions = useMemo(() => [...new Set(testingRuns.map((run) => run.method_type).filter(Boolean))].map((value) => facetOption(value, value, sourceFacetCounts.method, sourceMethodFilters)).sort((a, b) => a.label.localeCompare(b.label)), [sourceFacetCounts.method, sourceMethodFilters, testingRuns]);
  useEffect(() => setSourcePage(1), [sourceSearch, sourceModelFilters, sourceTrainingDatasetFilters, sourceDatasetFilters, sourceRoiFilters, sourceMetricFilters, sourceAggregationFilters, sourcePreprocessingFilters, sourceMethodFilters]);
  useEffect(() => setSourcePage((page) => Math.min(page, Math.max(1, Math.ceil(filteredRuns.length / DEFAULT_TABLE_PAGE_SIZE)))), [filteredRuns.length]);

  return (
      <Paper withBorder p="md">
        <Stack gap="md">
          <Group justify="space-between" wrap="wrap">
            <Text fw={700}>1. Select inference source{multiple ? 's' : ''}</Text>
            <Badge variant="light">{filteredRuns.length} matching inference{filteredRuns.length === 1 ? '' : 's'}</Badge>
          </Group>
          <Group justify="space-between" wrap="wrap">
            <Button variant="subtle" size="compact-sm" leftSection={<SlidersHorizontal size={16} />} rightSection={sourceFiltersOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />} onClick={() => setSourceFiltersOpen((open) => !open)}>Filters</Button>
            {(sourceSearch.trim() || sourceModelFilters.length || sourceTrainingDatasetFilters.length || sourceDatasetFilters.length || sourceRoiFilters.length || sourceMetricFilters.length || sourceAggregationFilters.length || sourcePreprocessingFilters.length || sourceMethodFilters.length) ? (
              <Button variant="subtle" color="gray" size="compact-sm" onClick={() => {
                setSourceSearch('');
                setSourceModelFilters([]);
                setSourceTrainingDatasetFilters([]);
                setSourceDatasetFilters([]);
                setSourceRoiFilters([]);
                setSourceMetricFilters([]);
                setSourceAggregationFilters([]);
                setSourcePreprocessingFilters([]);
                setSourceMethodFilters([]);
              }}>Reset filters</Button>
            ) : null}
          </Group>
          <Collapse in={sourceFiltersOpen}>
            <Stack gap="sm">
              <TextInput placeholder="Search inference, model, pipeline or dataset" leftSection={<Search size={16} />} value={sourceSearch} onChange={(event) => setSourceSearch(event.currentTarget.value)} />
              <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
                <MultiSelect label="Models" searchable clearable data={sourceModelOptions} value={sourceModelFilters} onChange={setSourceModelFilters} />
                <MultiSelect label="Training datasets" searchable clearable data={sourceTrainingDatasetOptions} value={sourceTrainingDatasetFilters} onChange={setSourceTrainingDatasetFilters} />
                <MultiSelect label="Inference datasets" searchable clearable data={sourceDatasetOptions} value={sourceDatasetFilters} onChange={setSourceDatasetFilters} />
                <MultiSelect label="ROI" searchable clearable data={sourceRoiOptions} value={sourceRoiFilters} onChange={setSourceRoiFilters} />
                <MultiSelect label="Metrics" searchable clearable data={sourceMetricOptions} value={sourceMetricFilters} onChange={setSourceMetricFilters} />
                <MultiSelect label="Score aggregation" searchable clearable data={sourceAggregationOptions} value={sourceAggregationFilters} onChange={setSourceAggregationFilters} />
                <MultiSelect label="Preprocessing" searchable clearable data={sourcePreprocessingOptions} value={sourcePreprocessingFilters} onChange={setSourcePreprocessingFilters} />
                <MultiSelect label="Methods" searchable clearable data={sourceMethodOptions} value={sourceMethodFilters} onChange={setSourceMethodFilters} />
              </SimpleGrid>
              <Text size="xs" c="dimmed">Counts show finished inferences available after the search and all other categories. Multiple values inside one category use OR.</Text>
            </Stack>
          </Collapse>
          {multiple && <Group>
            <Button variant="light" size="compact-sm" disabled={disabled || !filteredRuns.length} onClick={() => onSelectionChange([...new Set([...selectedIds, ...filteredRuns.map((run) => String(run.id))])])}>Select all filtered</Button>
            <Button variant="subtle" size="compact-sm" disabled={disabled || !selectedIds.length} onClick={() => onSelectionChange([])}>Clear selection</Button>
            <Badge>{selectedIds.length} selected</Badge>
          </Group>}
          <ScrollArea.Autosize mah={360} type="auto" offsetScrollbars>
            <Table striped highlightOnHover miw={1040}>
              <Table.Thead><Table.Tr>{multiple && <Table.Th>Select</Table.Th>}<Table.Th>Inference</Table.Th><Table.Th>Model</Table.Th><Table.Th>Dataset</Table.Th><Table.Th>ROI</Table.Th><Table.Th>Score configuration</Table.Th><Table.Th>Frames</Table.Th><Table.Th>Finished</Table.Th>{!multiple && <Table.Th />}</Table.Tr></Table.Thead>
              <Table.Tbody>
                {pagedFilteredRuns.map((run) => {
                  const selected = selectedIds.includes(String(run.id));
                  return <Table.Tr key={run.id} bg={selected ? 'var(--mantine-color-green-light)' : undefined}>
                    {multiple && <Table.Td><Checkbox aria-label={`Select ${run.name} (#${run.id})`} checked={selected} disabled={disabled} onChange={() => onSelectionChange(selected ? selectedIds.filter((id) => id !== String(run.id)) : [...selectedIds, String(run.id)])} /></Table.Td>}
                    <Table.Td><Text fw={selected ? 700 : 500}>{run.name}</Text></Table.Td>
                    <Table.Td>{run.training_pipeline_name || run.training_run_name || `Training run #${run.training_run_id}`}</Table.Td>
                    <Table.Td>{run.training_dataset_name || `Inference dataset #${run.training_dataset_id}`}</Table.Td>
                    <Table.Td>{roiLabelForRun(run)}</Table.Td>
                    <Table.Td>{metricLabel(metricKeyForRun(run))} · {aggregationLabel(aggregationKeyForRun(run))}</Table.Td>
                    <Table.Td>{(run.image_count ?? 0).toLocaleString()}</Table.Td>
                    <Table.Td>{formatTimestamp(run.ended_at)}</Table.Td>
                    {!multiple && <Table.Td><Button size="compact-sm" variant={selected ? 'filled' : 'light'} color={selected ? 'green' : 'blue'} disabled={disabled} onClick={() => onSelectionChange([String(run.id)])}>{selected ? 'Selected' : 'Use'}</Button></Table.Td>}
                  </Table.Tr>;
                })}
                {filteredRuns.length === 0 && <Table.Tr><Table.Td colSpan={8}><Stack align="center" gap="xs" py="md"><Text size="sm" c="dimmed">No finished inference matches the combined filters.</Text><Button variant="light" size="compact-sm" onClick={() => { setSourceSearch(''); setSourceModelFilters([]); setSourceTrainingDatasetFilters([]); setSourceDatasetFilters([]); setSourceRoiFilters([]); setSourceMetricFilters([]); setSourceAggregationFilters([]); setSourcePreprocessingFilters([]); setSourceMethodFilters([]); }}>Reset filters</Button></Stack></Table.Td></Table.Tr>}
              </Table.Tbody>
            </Table>
          </ScrollArea.Autosize>
          <TablePagination totalItems={filteredRuns.length} page={sourcePage} onChange={setSourcePage} />
          {multiple && selectedIds.length > 0 && <Alert color="green" title="Selected inferences"><Group gap="xs">{selectedIds.map((id) => {
            const run = testingRunById.get(Number(id));
            return <Badge key={id} color={filteredRuns.some((item) => String(item.id) === id) ? 'green' : 'yellow'}>{run?.name ?? `Inference #${id}`} · #{id}{!filteredRuns.some((item) => String(item.id) === id) ? ' · Hidden by current filters' : ''}</Badge>;
          })}</Group></Alert>}
          {!multiple && selectedRun && (
            <Alert color="green" title="Selected inference">
              <Group gap="xs">
              <Badge variant="light">{selectedRun.name}</Badge>
              <Badge variant="light">{selectedRun.training_pipeline_name || selectedRun.training_run_name}</Badge>
              <Badge variant="light" color="teal">{selectedRun.training_dataset_name}</Badge>
              <Badge variant="light" color="violet">{roiLabelForRun(selectedRun)}</Badge>
              <Badge variant="light" color="gray">{selectedRun.image_count ?? 0} frames</Badge>
              {!filteredRuns.some((run) => run.id === selectedRun.id) && <Badge variant="light" color="yellow">Hidden by current filters</Badge>}
              </Group>
            </Alert>
          )}
        </Stack>
      </Paper>

  );
}
