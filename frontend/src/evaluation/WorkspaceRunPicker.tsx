import { Badge, Button, Collapse, Group, MultiSelect, Paper, ScrollArea, Select, SimpleGrid, Stack, Table, Text, TextInput } from '@mantine/core';
import { ChevronDown, ChevronRight, LineChart, Search, SlidersHorizontal } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

import { DEFAULT_TABLE_PAGE_SIZE, TablePagination } from '../components/TablePagination';
import {
  countFacetValues,
  facetOption,
  matchingFacetRecords,
  type FacetFilterState,
  type FacetRecord,
} from '../testing/facetFilters';
import {
  aggregationKeyForRun,
  aggregationLabel,
  metricKeyForRun,
  metricLabel,
  roiKeyForRun,
  roiLabelForRun,
} from '../testing/inferenceRunMetadata';
import type { TestingRun } from '../types';

/** Score series the workspace can evaluate for a run; ROI needs stored ROI scores. */
export function scoreSeriesOptions(run: TestingRun | null) {
  return [
    { value: 'score', label: 'Configured anomaly score' },
    { value: 'full_mse', label: 'Full-frame MSE' },
    ...(run?.roi_mse_mean != null ? [{ value: 'roi_mse', label: 'ROI score' }] : []),
  ];
}

function asFacetRecord(run: TestingRun): FacetRecord {
  return {
    id: String(run.id),
    facets: {
      dataset: [String(run.training_dataset_id)],
      roi: [roiKeyForRun(run)],
      metric: [metricKeyForRun(run)],
      aggregation: [aggregationKeyForRun(run)],
    },
    searchableValues: [
      `#${run.id}`,
      run.name,
      run.training_dataset_name,
      roiLabelForRun(run),
      metricLabel(metricKeyForRun(run)),
      aggregationLabel(aggregationKeyForRun(run)),
    ],
  };
}

function uniqueOptions(runs: TestingRun[], pair: (run: TestingRun) => [string, string]) {
  return [...new Map(runs.map(pair)).entries()];
}

/**
 * Inference run selection for a single model workspace, mirroring the table based
 * pickers on the analysis page.  Model, preprocessing and method are constant
 * inside one workspace, so only the run-level facets are offered.
 */
export function WorkspaceRunPicker({ runs, runId, score, onRun, onScore, onLoadPlot, loading, loaded }: {
  runs: TestingRun[];
  runId: number | null;
  score: string;
  onRun: (id: number | null) => void;
  onScore: (value: string) => void;
  onLoadPlot: () => void;
  loading: boolean;
  loaded: boolean;
}) {
  const [search, setSearch] = useState('');
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [dataset, setDataset] = useState<string[]>([]);
  const [roi, setRoi] = useState<string[]>([]);
  const [metric, setMetric] = useState<string[]>([]);
  const [aggregation, setAggregation] = useState<string[]>([]);
  const [page, setPage] = useState(1);
  const run = runs.find((item) => item.id === runId) ?? null;

  const records = useMemo(() => runs.map(asFacetRecord), [runs]);
  const filters = useMemo<FacetFilterState>(
    () => ({ query: search, selections: { dataset, roi, metric, aggregation } }),
    [aggregation, dataset, metric, roi, search],
  );
  const matchingIds = useMemo(
    () => new Set(matchingFacetRecords(records, filters).map((record) => record.id)),
    [filters, records],
  );
  const filtered = useMemo(() => runs.filter((item) => matchingIds.has(String(item.id))), [matchingIds, runs]);
  const counts = useMemo(() => ({
    dataset: countFacetValues(records, filters, 'dataset'),
    roi: countFacetValues(records, filters, 'roi'),
    metric: countFacetValues(records, filters, 'metric'),
    aggregation: countFacetValues(records, filters, 'aggregation'),
  }), [filters, records]);

  const datasetOptions = useMemo(() => uniqueOptions(runs, (item) => [String(item.training_dataset_id), item.training_dataset_name])
    .map(([value, label]) => facetOption(value, label, counts.dataset, dataset))
    .sort((left, right) => left.label.localeCompare(right.label)), [counts.dataset, dataset, runs]);
  const roiOptions = useMemo(() => uniqueOptions(runs, (item) => [roiKeyForRun(item), roiLabelForRun(item)])
    .map(([value, label]) => facetOption(value, label, counts.roi, roi))
    .sort((left, right) => left.label.localeCompare(right.label)), [counts.roi, roi, runs]);
  const metricOptions = useMemo(() => uniqueOptions(runs, (item) => [metricKeyForRun(item), metricLabel(metricKeyForRun(item))])
    .map(([value, label]) => facetOption(value, label, counts.metric, metric)), [counts.metric, metric, runs]);
  const aggregationOptions = useMemo(() => uniqueOptions(runs, (item) => [aggregationKeyForRun(item), aggregationLabel(aggregationKeyForRun(item))])
    .map(([value, label]) => facetOption(value, label, counts.aggregation, aggregation)), [aggregation, counts.aggregation, runs]);

  const paged = useMemo(
    () => filtered.slice((page - 1) * DEFAULT_TABLE_PAGE_SIZE, page * DEFAULT_TABLE_PAGE_SIZE),
    [filtered, page],
  );
  const activeFilters = Boolean(search.trim()) || dataset.length > 0 || roi.length > 0 || metric.length > 0 || aggregation.length > 0;
  const reset = () => { setSearch(''); setDataset([]); setRoi([]); setMetric([]); setAggregation([]); };

  useEffect(() => setPage(1), [search, dataset, roi, metric, aggregation]);
  useEffect(() => {
    setPage((current) => Math.min(current, Math.max(1, Math.ceil(filtered.length / DEFAULT_TABLE_PAGE_SIZE))));
  }, [filtered.length]);

  return (
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
          <Badge variant="light">{filtered.length} matching run{filtered.length === 1 ? '' : 's'}</Badge>
        </Group>
        {activeFilters && <Button variant="subtle" color="gray" size="compact-sm" onClick={reset}>Reset filters</Button>}
      </Group>
      <Collapse in={filtersOpen}>
        <Stack gap="sm">
          <TextInput
            placeholder="Search inference runs"
            leftSection={<Search size={16} />}
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
          />
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }}>
            <MultiSelect label="Inference datasets" data={datasetOptions} value={dataset} onChange={setDataset} searchable clearable />
            <MultiSelect label="ROI" data={roiOptions} value={roi} onChange={setRoi} searchable clearable />
            <MultiSelect label="Metrics" data={metricOptions} value={metric} onChange={setMetric} searchable clearable />
            <MultiSelect label="Score aggregation" data={aggregationOptions} value={aggregation} onChange={setAggregation} searchable clearable />
          </SimpleGrid>
          <Text size="xs" c="dimmed">Counts show the runs that would remain after the search and all other filter categories. Values inside one category use OR.</Text>
        </Stack>
      </Collapse>
      <Paper withBorder radius="sm">
        <ScrollArea>
          <Table striped highlightOnHover verticalSpacing="sm" miw={900}>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Run</Table.Th>
                <Table.Th>Inference dataset</Table.Th>
                <Table.Th>ROI</Table.Th>
                <Table.Th>Metric / aggregation</Table.Th>
                <Table.Th>Images</Table.Th>
                <Table.Th>Finished</Table.Th>
                <Table.Th />
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {paged.map((item) => {
                const selected = item.id === runId;
                return (
                  <Table.Tr key={item.id}>
                    <Table.Td>#{item.id} · {item.name}</Table.Td>
                    <Table.Td>{item.training_dataset_name}</Table.Td>
                    <Table.Td><Badge size="xs" variant="light" color={item.roi_id === null ? 'gray' : 'teal'}>{roiLabelForRun(item)}</Badge></Table.Td>
                    <Table.Td>
                      <Group gap={4}>
                        <Badge size="xs" variant="light" color="blue">{metricLabel(metricKeyForRun(item))}</Badge>
                        <Badge size="xs" variant="light" color="violet">{aggregationLabel(aggregationKeyForRun(item))}</Badge>
                      </Group>
                    </Table.Td>
                    <Table.Td>{item.image_count ?? '—'}</Table.Td>
                    <Table.Td>{item.ended_at ? item.ended_at.replace('T', ' ').slice(0, 19) : '—'}</Table.Td>
                    <Table.Td>
                      <Group justify="flex-end">
                        <Button
                          size="compact-sm"
                          variant={selected ? 'filled' : 'light'}
                          color={selected ? 'green' : 'blue'}
                          onClick={() => onRun(selected ? null : item.id)}
                        >
                          {selected ? 'Selected' : 'Use'}
                        </Button>
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                );
              })}
              {filtered.length === 0 && (
                <Table.Tr>
                  <Table.Td colSpan={7}>
                    <Stack align="center" gap="xs" py="md">
                      <Text size="sm" c="dimmed">
                        {runs.length === 0 ? 'This model artifact has no finished inference run yet.' : 'No finished inference run matches the combined filters.'}
                      </Text>
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
      <SimpleGrid cols={{ base: 1, md: 2 }}>
        <Select
          label="Score series"
          data={scoreSeriesOptions(run)}
          value={score}
          disabled={!run}
          onChange={(value) => onScore(value ?? 'score')}
        />
        <Button
          mt={{ base: 0, md: 25 }}
          leftSection={<LineChart size={16} />}
          disabled={!run}
          loading={loading}
          onClick={onLoadPlot}
        >
          {loading ? 'Loading score plot…' : loaded ? 'Reload score plot' : 'Load score plot'}
        </Button>
      </SimpleGrid>
    </Stack>
  );
}
