import { Badge, Button, Collapse, Group, MultiSelect, Select, SimpleGrid, Stack, Text, TextInput } from '@mantine/core';
import { ChevronDown, ChevronRight, Search, SlidersHorizontal, X } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { TestingRun } from '../types';
import {
  aggregationKeyForRun,
  aggregationLabel,
  metricKeyForRun,
  metricLabel,
  roiKeyForRun,
  roiLabelForRun,
} from '../testing/inferenceRunMetadata';
import {
  emptyEvaluationRunFilters,
  evaluationRunsCompatible,
  evaluationRunMatches,
  filterEvaluationRuns,
  type EvaluationRunFilters,
} from './helpers';

function uniqueOptions(values: Array<[string, string]>): Array<{ value: string; label: string }> {
  const labels = new Map<string, string>();
  values.forEach(([value, label]) => labels.set(value, label));
  return [...labels].map(([value, label]) => ({ value, label })).sort((left, right) => left.label.localeCompare(right.label));
}

function omitFacet(filters: EvaluationRunFilters, facet: keyof EvaluationRunFilters): EvaluationRunFilters {
  return { ...filters, [facet]: facet === 'query' ? '' : [] } as EvaluationRunFilters;
}

function countedOptions(
  runs: TestingRun[],
  filters: EvaluationRunFilters,
  facet: Exclude<keyof EvaluationRunFilters, 'query'>,
  pairs: (run: TestingRun) => Array<[string, string]>,
): Array<{ value: string; label: string; disabled: boolean }> {
  const baseOptions = uniqueOptions(runs.flatMap(pairs));
  const selected = filters[facet];
  const candidates = runs.filter((run) => evaluationRunMatches(run, omitFacet(filters, facet)));
  return baseOptions.map((option) => {
    const count = candidates.filter((run) => pairs(run).some(([value]) => value === option.value)).length;
    return {
      ...option,
      label: `${option.label} (${count})`,
      disabled: count === 0 && !selected.includes(option.value),
    };
  });
}

export function EvaluationRunPicker({
  runs,
  value,
  onChange,
  label,
  description,
  compatibleWith,
  disabled = false,
}: {
  runs: TestingRun[];
  value: number | null;
  onChange: (runId: number | null) => void;
  label: string;
  description?: string;
  compatibleWith?: TestingRun | null;
  disabled?: boolean;
}) {
  const [filters, setFilters] = useState<EvaluationRunFilters>(emptyEvaluationRunFilters);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const compatibleRuns = useMemo(
    () => runs.filter((run) => compatibleWith == null || evaluationRunsCompatible(compatibleWith, run)),
    [compatibleWith, runs],
  );
  const matchingRuns = useMemo(() => filterEvaluationRuns(compatibleRuns, filters), [compatibleRuns, filters]);
  const selectedRun = runs.find((run) => run.id === value);
  const selectableRuns = selectedRun && !matchingRuns.some((run) => run.id === selectedRun.id)
    ? [selectedRun, ...matchingRuns]
    : matchingRuns;
  const activeCount = Object.entries(filters)
    .filter(([key, selected]) => key === 'query' ? Boolean(String(selected).trim()) : (selected as string[]).length > 0)
    .length;

  const options = useMemo(() => ({
    model: countedOptions(compatibleRuns, filters, 'model', (run) => [[String(run.training_run_id), run.training_pipeline_name || run.training_run_name]]),
    modelTrainingDataset: countedOptions(compatibleRuns, filters, 'modelTrainingDataset', (run) => run.model_training_dataset_names.map((name) => [name, name])),
    inferenceDataset: countedOptions(compatibleRuns, filters, 'inferenceDataset', (run) => [[String(run.training_dataset_id), run.training_dataset_name]]),
    roi: countedOptions(compatibleRuns, filters, 'roi', (run) => [[roiKeyForRun(run), roiLabelForRun(run)]]),
    metric: countedOptions(compatibleRuns, filters, 'metric', (run) => [[metricKeyForRun(run), metricLabel(metricKeyForRun(run))]]),
    aggregation: countedOptions(compatibleRuns, filters, 'aggregation', (run) => [[aggregationKeyForRun(run), aggregationLabel(aggregationKeyForRun(run))]]),
    preprocessing: countedOptions(compatibleRuns, filters, 'preprocessing', (run) => [[run.preprocessing_pipeline_name, run.preprocessing_pipeline_name]]),
    method: countedOptions(compatibleRuns, filters, 'method', (run) => [[run.method_type, run.method_type.replaceAll('_', ' ')]]),
  }), [compatibleRuns, filters]);

  function setFacet(facet: Exclude<keyof EvaluationRunFilters, 'query'>, values: string[]) {
    setFilters((current) => ({ ...current, [facet]: values }));
  }

  return (
    <Stack gap="xs">
      <Group justify="space-between" align="end">
        <Group gap="xs">
          <Button
            variant="default"
            size="compact-sm"
            leftSection={<SlidersHorizontal size={14} />}
            rightSection={filtersOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            onClick={() => setFiltersOpen((current) => !current)}
            disabled={disabled}
          >
            Filters{activeCount ? ` (${activeCount})` : ''}
          </Button>
          <Badge variant="light">{matchingRuns.length} matching runs</Badge>
        </Group>
        {activeCount > 0 && (
          <Button variant="subtle" color="gray" size="compact-sm" leftSection={<X size={13} />} onClick={() => setFilters(emptyEvaluationRunFilters())}>
            Reset
          </Button>
        )}
      </Group>
      <Collapse in={filtersOpen}>
        <Stack gap="xs">
          <TextInput
            leftSection={<Search size={15} />}
            placeholder="Search model, pipeline, dataset or inference run"
            value={filters.query}
            onChange={(event) => setFilters((current) => ({ ...current, query: event.currentTarget.value }))}
            disabled={disabled}
          />
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="xs">
            <MultiSelect label="Models" data={options.model} value={filters.model} onChange={(values) => setFacet('model', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="Training datasets" data={options.modelTrainingDataset} value={filters.modelTrainingDataset} onChange={(values) => setFacet('modelTrainingDataset', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="Inference datasets" data={options.inferenceDataset} value={filters.inferenceDataset} onChange={(values) => setFacet('inferenceDataset', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="ROI" data={options.roi} value={filters.roi} onChange={(values) => setFacet('roi', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="Metrics" data={options.metric} value={filters.metric} onChange={(values) => setFacet('metric', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="Aggregation" data={options.aggregation} value={filters.aggregation} onChange={(values) => setFacet('aggregation', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="Preprocessing" data={options.preprocessing} value={filters.preprocessing} onChange={(values) => setFacet('preprocessing', values)} searchable clearable disabled={disabled} />
            <MultiSelect label="Method" data={options.method} value={filters.method} onChange={(values) => setFacet('method', values)} searchable clearable disabled={disabled} />
          </SimpleGrid>
          <Text size="xs" c="dimmed">Values within one filter use OR; filter groups are combined with AND.</Text>
        </Stack>
      </Collapse>
      <Select
        label={label}
        description={description}
        placeholder="Select a finished inference run"
        data={selectableRuns.map((run) => ({
          value: String(run.id),
          label: `${run.name} · ${run.training_dataset_name} · ${metricLabel(metricKeyForRun(run))}/${aggregationLabel(aggregationKeyForRun(run))}`,
        }))}
        value={value == null ? null : String(value)}
        onChange={(selected) => onChange(selected ? Number(selected) : null)}
        searchable
        clearable
        disabled={disabled}
        nothingFoundMessage="No compatible finished inference run matches the filters"
      />
    </Stack>
  );
}
