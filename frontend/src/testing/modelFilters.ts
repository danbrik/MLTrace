import { countFacetValues, facetRecordMatches, type FacetFilterState, type FacetRecord } from './facetFilters';

export type ModelFilterFacet = 'dataset' | 'preprocessing' | 'method';

export type ModelFilterMetadata = {
  datasetIds: string[];
  preprocessingId: string | null;
  methodId: string | null;
  inputResolution: string | null;
  searchableValues: string[];
};

export type ModelFilterState = {
  query: string;
  datasetIds: string[];
  preprocessingIds: string[];
  methodIds: string[];
  requiredInputResolution: string | null;
};

export function modelMatchesFilters(
  record: ModelFilterMetadata,
  filters: ModelFilterState,
  ignoredFacet?: ModelFilterFacet,
): boolean {
  if (filters.requiredInputResolution && record.inputResolution !== filters.requiredInputResolution) return false;
  return facetRecordMatches(asFacetRecord(record), asFacetState(filters), ignoredFacet);
}

export function countModelFacetValues(
  records: ModelFilterMetadata[],
  filters: ModelFilterState,
  facet: ModelFilterFacet,
): Map<string, number> {
  const eligible = filters.requiredInputResolution
    ? records.filter((record) => record.inputResolution === filters.requiredInputResolution)
    : records;
  return countFacetValues(
    eligible.map((record, index) => ({ ...asFacetRecord(record), id: String(index) })),
    asFacetState(filters),
    facet,
  );
}

function asFacetRecord(record: ModelFilterMetadata): FacetRecord {
  return {
    id: record.searchableValues.join('\u0000'),
    facets: {
      dataset: record.datasetIds,
      preprocessing: record.preprocessingId ? [record.preprocessingId] : [],
      method: record.methodId ? [record.methodId] : [],
    },
    searchableValues: record.searchableValues,
  };
}

function asFacetState(filters: ModelFilterState): FacetFilterState {
  return {
    query: filters.query,
    selections: {
      dataset: filters.datasetIds,
      preprocessing: filters.preprocessingIds,
      method: filters.methodIds,
    },
  };
}
