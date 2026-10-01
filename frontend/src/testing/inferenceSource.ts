import type { TestingRun } from '../types';
import type { FacetRecord } from './facetFilters';
import { aggregationKeyForRun, aggregationLabel, metricKeyForRun, metricLabel, roiKeyForRun, roiLabelForRun } from './inferenceRunMetadata';

export function inferenceFacetRecord(run: TestingRun): FacetRecord {
  return {
    id: String(run.id),
    facets: {
      model: [String(run.training_run_id)],
      trainingDataset: run.model_training_dataset_names ?? [],
      dataset: [String(run.training_dataset_id)],
      roi: [roiKeyForRun(run)],
      metric: [metricKeyForRun(run)],
      aggregation: [aggregationKeyForRun(run)],
      preprocessing: run.preprocessing_pipeline_name ? [run.preprocessing_pipeline_name] : [],
      method: run.method_type ? [run.method_type] : [],
    },
    searchableValues: [
      run.name,
      run.training_run_name,
      run.training_pipeline_name,
      ...(run.model_training_dataset_names ?? []),
      run.training_dataset_name,
      roiLabelForRun(run),
      run.preprocessing_pipeline_name,
      run.method_type,
      metricLabel(metricKeyForRun(run)),
      aggregationLabel(aggregationKeyForRun(run)),
    ].filter(Boolean),
  };
}

