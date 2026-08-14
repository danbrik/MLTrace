export type FacetRecord = {
  id: string;
  groupId?: string;
  facets: Record<string, string[]>;
  searchableValues: string[];
};

export type FacetFilterState = {
  query: string;
  selections: Record<string, string[]>;
};

export function facetRecordMatches(
  record: FacetRecord,
  filters: FacetFilterState,
  ignoredFacet?: string,
): boolean {
  const query = filters.query.trim().toLowerCase();
  if (query && !record.searchableValues.some((value) => value.toLowerCase().includes(query))) return false;

  return Object.entries(filters.selections).every(([facet, selected]) => {
    if (facet === ignoredFacet || selected.length === 0) return true;
    const values = record.facets[facet] ?? [];
    return values.some((value) => selected.includes(value));
  });
}

export function matchingFacetRecords<T extends FacetRecord>(records: T[], filters: FacetFilterState): T[] {
  return records.filter((record) => facetRecordMatches(record, filters));
}

export function matchingFacetGroupIds(records: FacetRecord[], filters: FacetFilterState): Set<string> {
  return new Set(
    records
      .filter((record) => facetRecordMatches(record, filters))
      .map((record) => record.groupId ?? record.id),
  );
}

export function countFacetValues(
  records: FacetRecord[],
  filters: FacetFilterState,
  facet: string,
): Map<string, number> {
  const groupsByValue = new Map<string, Set<string>>();
  records.forEach((record) => {
    if (!facetRecordMatches(record, filters, facet)) return;
    const groupId = record.groupId ?? record.id;
    new Set(record.facets[facet] ?? []).forEach((value) => {
      const groups = groupsByValue.get(value) ?? new Set<string>();
      groups.add(groupId);
      groupsByValue.set(value, groups);
    });
  });
  return new Map([...groupsByValue.entries()].map(([value, groups]) => [value, groups.size]));
}

export function facetOption(
  value: string,
  label: string,
  counts: Map<string, number>,
  selected: string[],
) {
  const count = counts.get(value) ?? 0;
  return {
    value,
    label: `${label} (${count})`,
    disabled: count === 0 && !selected.includes(value),
  };
}
